"""
Evaluate SAMSum LoRA checkpoints produced by model_lora2.py.

Examples:
  python eval_lora_samsum.py --rank 8
  python eval_lora_samsum.py --checkpoint output/diffugpt-m-lora/samsum-r8-lr5e-05-bs16/lora_final.pt --rank 8

Defaults match eval_full2.py:
  - SAMSum test split
  - ROUGE-L and BERTScore
  - 64 diffusion steps, 128 max generated target tokens, logits temp 0.95, top-p 0.9
"""

import argparse
import glob
import json
from pathlib import Path

import torch

from model_full2 import compute_rouge_l
from model_lora2 import (
    build_lora_diffugpt,
    generate_target_text,
    load_diffugpt,
    parse_target_layers,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def build_samsum_prompt(dialogue: str) -> str:
    return f"Summarize the following dialogue.\n\nDialogue:\n{dialogue}\n\nSummary:"


def parse_metric_list(value: str):
    metrics = [item.strip().lower() for item in value.split(",") if item.strip()]
    aliases = {"rougel": "rougeL", "rouge_l": "rougeL", "bertscore": "bertscore"}
    parsed = []
    for metric in metrics:
        if metric not in aliases:
            raise argparse.ArgumentTypeError(
                f"Unsupported SAMSum metric '{metric}'. Use rougeL, bertscore, or both."
            )
        normalized = aliases[metric]
        if normalized not in parsed:
            parsed.append(normalized)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one SAMSum metric is required.")
    return parsed


def compute_bertscore(predictions, references, model_type: str):
    try:
        import evaluate
    except ImportError:
        print("evaluate is not installed; skipping BERTScore metric.")
        return {}

    bertscore = evaluate.load("bertscore")
    scores = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type=model_type,
    )

    def mean(values):
        return 100.0 * sum(float(value) for value in values) / max(1, len(values))

    return {
        "bertscore_precision": mean(scores["precision"]),
        "bertscore_recall": mean(scores["recall"]),
        "bertscore_f1": mean(scores["f1"]),
    }


def read_checkpoint(path: str):
    payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    return state_dict, metadata


def find_checkpoint(args):
    if args.checkpoint:
        return args.checkpoint
    if args.rank is None:
        raise ValueError("--rank is required when --checkpoint is not provided.")

    pattern = str(Path(args.lora_root) / f"samsum-r{args.rank}-*" / "lora_final.pt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No LoRA checkpoint found for rank {args.rank} with pattern: {pattern}"
        )
    return max(matches, key=lambda path: Path(path).stat().st_mtime)


def default_output_prefix(checkpoint_path: str) -> str:
    checkpoint = Path(checkpoint_path)
    parent = checkpoint.parent.name
    if parent:
        return f"{parent}_{checkpoint.stem}"
    return checkpoint.stem


def apply_metadata_defaults(args, metadata: dict):
    if args.model_name is None:
        args.model_name = metadata.get("model_name", "diffusionfamily/diffugpt-m")
    if args.base_model_name is None:
        args.base_model_name = metadata.get("base_model_name", "gpt2-medium")
    if args.load_mode is None:
        args.load_mode = metadata.get("load_mode", "ddm")
    if args.rank is None:
        args.rank = metadata.get("rank")
    if args.rank is None:
        raise ValueError("--rank is required when checkpoint metadata does not include rank.")
    if args.lora_alpha is None:
        args.lora_alpha = metadata.get("lora_alpha", 2 * args.rank)
    if args.lora_dropout is None:
        args.lora_dropout = metadata.get("lora_dropout", 0.05)
    if args.target_layers is None:
        args.target_layers = metadata.get("target_layers")
    if args.train_embeddings is None:
        args.train_embeddings = bool(metadata.get("train_embeddings", True))
    if args.max_len is None:
        args.max_len = int(metadata.get("max_len", 256))
    if args.shift is None:
        args.shift = bool(metadata.get("shift", True))


def build_model(args, state_dict, device):
    from transformers import AutoConfig, AutoTokenizer

    try:
        import transformers.models.llama.modeling_llama as _llama

        if not hasattr(_llama, "LlamaFlashAttention2"):
            _llama.LlamaFlashAttention2 = _llama.LlamaAttention
    except ImportError:
        pass

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer has no mask token. Expected a DiffuGPT tokenizer.")

    ddm = load_diffugpt(args, tokenizer=tokenizer, config=config, device=device)
    ddm = build_lora_diffugpt(
        ddm,
        r=args.rank,
        lora_alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_layers=parse_target_layers(args.target_layers),
        train_embeddings=args.train_embeddings,
    )

    missing, unexpected = ddm.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing keys when loading LoRA checkpoint")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys when loading LoRA checkpoint")

    ddm = ddm.to(device)
    ddm.eval()
    return ddm, tokenizer


def load_samsum(split: str):
    from datasets import load_dataset

    return load_dataset("knkarthick/samsum", split=split)


def evaluate(model, tokenizer, args, device):
    ds = load_samsum(args.split)
    if args.max_examples and args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{args.output_prefix}_samsum_predictions.jsonl"

    predictions = []
    references = []

    print(f"\nEvaluating samsum split={args.split} examples={len(ds)}")
    with pred_path.open("w", encoding="utf-8") as f:
        with torch.no_grad():
            iterator = ds
            if tqdm is not None:
                iterator = tqdm(ds, total=len(ds), desc="Evaluating samsum")

            for idx, example in enumerate(iterator):
                prompt = build_samsum_prompt(example["dialogue"])
                target = example["summary"]
                prediction = generate_target_text(
                    model,
                    tokenizer,
                    args,
                    prompt=prompt,
                    target=target,
                    device=device,
                )

                predictions.append(prediction)
                references.append(target)

                row = {
                    "index": idx,
                    "dataset": "samsum",
                    "split": args.split,
                    "dialogue": example["dialogue"],
                    "prompt": prompt,
                    "reference": target,
                    "prediction": prediction,
                }
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                f.flush()

                if idx < args.print_examples:
                    print(f"\n--- samsum example {idx} ---")
                    print(f"Ref:  {target[:300]}")
                    print(f"Pred: {prediction[:300]}")

                if (idx + 1) % args.log_every == 0:
                    print(f"[samsum] processed {idx + 1}/{len(ds)}", flush=True)

    metrics = {"dataset": "samsum", "split": args.split, "examples": len(predictions)}
    if "rougeL" in args.samsum_metrics:
        rouge_l = compute_rouge_l(predictions, references)
        metrics["rougeL"] = rouge_l
        if rouge_l is not None:
            print(f"samsum: ROUGE-L {rouge_l:.2f}")
        else:
            print("samsum: ROUGE-L unavailable")

    if "bertscore" in args.samsum_metrics:
        bertscore_metrics = compute_bertscore(
            predictions,
            references,
            model_type=args.bertscore_model_type,
        )
        metrics.update(bertscore_metrics)
        if bertscore_metrics:
            print(
                f"samsum: BERTScore F1 {bertscore_metrics['bertscore_f1']:.2f} "
                f"(P {bertscore_metrics['bertscore_precision']:.2f}, "
                f"R {bertscore_metrics['bertscore_recall']:.2f})"
            )

    metrics["predictions_file"] = str(pred_path)
    print(f"Wrote predictions to {pred_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--lora_root", type=str, default="output/diffugpt-m-lora")

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--base_model_name", type=str, default=None)
    parser.add_argument("--load_mode", choices=["ddm", "causal_lm"], default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--target_layers", type=str, default=None)
    parser.add_argument("--train_embeddings", action="store_true", default=None)
    parser.add_argument("--no_train_embeddings", action="store_false", dest="train_embeddings")

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_examples", type=int, default=0, help="0 means evaluate the full split.")
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--metric_diffusion_steps", type=int, default=64)
    parser.add_argument("--metric_max_new_tokens", type=int, default=128)
    parser.add_argument("--metric_logits_temp", type=float, default=0.95)
    parser.add_argument("--metric_topp_temp", type=float, default=0.9)
    parser.add_argument(
        "--samsum_metrics",
        type=parse_metric_list,
        default=parse_metric_list("rougeL,bertscore"),
        help="Comma-separated SAMSum metrics: rougeL, bertscore, or both.",
    )
    parser.add_argument("--bertscore_model_type", type=str, default="microsoft/deberta-xlarge-mnli")
    parser.add_argument("--shift", action="store_true", default=None)
    parser.add_argument("--no_shift", action="store_false", dest="shift")

    parser.add_argument("--output_dir", type=str, default="evaluation/lora_samsum_results")
    parser.add_argument("--output_prefix", type=str, default=None)
    parser.add_argument("--print_examples", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    args.checkpoint = find_checkpoint(args)
    state_dict, metadata = read_checkpoint(args.checkpoint)
    apply_metadata_defaults(args, metadata)
    if args.output_prefix is None:
        args.output_prefix = default_output_prefix(args.checkpoint)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device} | checkpoint={args.checkpoint} | rank={args.rank}")
    print(
        "Decoding: "
        f"steps={args.metric_diffusion_steps}, max_new_tokens={args.metric_max_new_tokens}, "
        f"logits_temp={args.metric_logits_temp}, top_p={args.metric_topp_temp}, "
        f"max_len={args.max_len}, shift={args.shift}"
    )

    model, tokenizer = build_model(args, state_dict, device)
    del state_dict

    metrics = evaluate(model, tokenizer, args, device)
    all_metrics = {
        "checkpoint": args.checkpoint,
        "metadata": metadata,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_layers": args.target_layers,
        "train_embeddings": args.train_embeddings,
        "decoding": {
            "metric_diffusion_steps": args.metric_diffusion_steps,
            "metric_max_new_tokens": args.metric_max_new_tokens,
            "metric_logits_temp": args.metric_logits_temp,
            "metric_topp_temp": args.metric_topp_temp,
            "max_len": args.max_len,
            "shift": args.shift,
            "samsum_metrics": args.samsum_metrics,
            "bertscore_model_type": args.bertscore_model_type,
        },
        "results": {"samsum": metrics},
    }

    metrics_path = Path(args.output_dir) / f"{args.output_prefix}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=True)

    print(f"\nWrote metrics to {metrics_path}")
    print(json.dumps(all_metrics["results"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
