"""
Evaluate full-finetuned DiffuGPT checkpoints produced by model_full2.py.

Examples:
  python eval_full2.py --checkpoint output/diffugpt-m-full/samsum_seed0/full_final.pt --trained_dataset samsum
  python eval_full2.py --checkpoint output/diffugpt-m-full/gsm8k_seed0/full_final.pt --trained_dataset gsm8k

Defaults:
  - SAMSum-trained checkpoints evaluate on SAMSum.
  - GSM8K-trained checkpoints evaluate on GSM8K and arithmetic.
  - Decoding defaults match model_full2.py metric evals.
"""

import argparse
import json
import re
from pathlib import Path

import torch

from model_full2 import compute_rouge_l, generate_target_text, load_diffugpt

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


ARITHMETIC_SUBTASKS = [
    "arithmetic__add_or_sub",
    "arithmetic__add_sub_multiple",
    "arithmetic__mul",
]


def build_samsum_prompt(dialogue: str) -> str:
    return f"Summarize the following dialogue.\n\nDialogue:\n{dialogue}\n\nSummary:"


def build_gsm8k_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def build_arithmetic_prompt(question: str) -> str:
    return f"{question}\nAnswer:"


def make_prompt_target(example: dict, dataset_name: str):
    if dataset_name == "samsum":
        return build_samsum_prompt(example["dialogue"]), example["summary"]
    if dataset_name == "gsm8k":
        return build_gsm8k_prompt(example["question"]), example["answer"]
    if dataset_name == "arithmetic":
        return build_arithmetic_prompt(example["question"]), example["answer"]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def default_eval_split(dataset_name: str) -> str:
    if dataset_name == "samsum":
        return "validation"
    if dataset_name in ("gsm8k", "arithmetic"):
        return "test"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def default_eval_datasets(trained_dataset: str):
    if trained_dataset == "samsum":
        return ["samsum"]
    if trained_dataset == "gsm8k":
        return ["gsm8k", "arithmetic"]
    raise ValueError(f"Unsupported trained dataset: {trained_dataset}")


def load_eval_dataset(dataset_name: str, split: str, arithmetic_samples_per_subtask: int):
    from datasets import concatenate_datasets, load_dataset

    if dataset_name == "samsum":
        return load_dataset("knkarthick/samsum", split=split)
    if dataset_name == "gsm8k":
        return load_dataset("openai/gsm8k", "main", split=split)
    if dataset_name == "arithmetic":
        parts = []
        for subtask in ARITHMETIC_SUBTASKS:
            split_str = (
                f"{split}[:{arithmetic_samples_per_subtask}]"
                if arithmetic_samples_per_subtask > 0
                else split
            )
            parts.append(
                load_dataset(
                    "deepmind/math_dataset",
                    subtask,
                    split=split_str,
                    trust_remote_code=True,
                )
            )
        return concatenate_datasets(parts).shuffle(seed=42)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def extract_numeric_answer(text: str) -> str:
    after_marker = re.search(r"####\s*([-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?)", text)
    if after_marker:
        return after_marker.group(1).replace(",", "")

    numbers = re.findall(r"[-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?", text)
    if not numbers:
        return ""
    return numbers[-1].replace(",", "")


def parse_dataset_list(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def read_checkpoint(path: str):
    payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    return state_dict, metadata


def apply_metadata_defaults(args, metadata: dict):
    if args.model_name is None:
        args.model_name = metadata.get("model_name", "diffusionfamily/diffugpt-m")
    if args.base_model_name is None:
        args.base_model_name = metadata.get("base_model_name", "gpt2-medium")
    if args.load_mode is None:
        args.load_mode = metadata.get("load_mode", "ddm")
    if args.max_len is None:
        args.max_len = int(metadata.get("max_len", 256))
    if args.shift is None:
        args.shift = bool(metadata.get("shift", True))
    if args.trained_dataset is None:
        args.trained_dataset = metadata.get("dataset")
    if args.trained_dataset is None:
        raise ValueError("--trained_dataset is required when checkpoint metadata does not include dataset.")


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
    missing, unexpected = ddm.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing keys when loading checkpoint")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys when loading checkpoint")

    ddm = ddm.to(device)
    ddm.eval()
    return ddm, tokenizer


def raw_example_payload(example: dict, dataset_name: str):
    if dataset_name == "samsum":
        return {"dialogue": example["dialogue"]}
    if dataset_name in ("gsm8k", "arithmetic"):
        return {"question": example["question"]}
    return {}


def evaluate_dataset(model, tokenizer, args, dataset_name: str, device):
    split = args.split or default_eval_split(dataset_name)
    ds = load_eval_dataset(dataset_name, split, args.arithmetic_samples_per_subtask)
    if args.max_examples and args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{args.output_prefix}_{dataset_name}_predictions.jsonl"

    predictions = []
    references = []
    correct = 0

    print(f"\nEvaluating {dataset_name} split={split} examples={len(ds)}")
    with pred_path.open("w", encoding="utf-8") as f:
        with torch.no_grad():
            iterator = ds
            if tqdm is not None:
                iterator = tqdm(ds, total=len(ds), desc=f"Evaluating {dataset_name}")

            for idx, example in enumerate(iterator):
                prompt, target = make_prompt_target(example, dataset_name)
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
                    "dataset": dataset_name,
                    "split": split,
                    "prompt": prompt,
                    "reference": target,
                    "prediction": prediction,
                    **raw_example_payload(example, dataset_name),
                }

                if dataset_name in ("gsm8k", "arithmetic"):
                    pred_answer = extract_numeric_answer(prediction)
                    ref_answer = extract_numeric_answer(target)
                    row["pred_answer"] = pred_answer
                    row["ref_answer"] = ref_answer
                    row["correct"] = pred_answer == ref_answer
                    correct += int(row["correct"])

                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                f.flush()

                if idx < args.print_examples:
                    print(f"\n--- {dataset_name} example {idx} ---")
                    print(f"Ref:  {target[:300]}")
                    print(f"Pred: {prediction[:300]}")

                if (idx + 1) % args.log_every == 0:
                    print(f"[{dataset_name}] processed {idx + 1}/{len(ds)}", flush=True)

    metrics = {"dataset": dataset_name, "split": split, "examples": len(predictions)}
    if dataset_name == "samsum":
        rouge_l = compute_rouge_l(predictions, references)
        metrics["rougeL"] = rouge_l
        if rouge_l is not None:
            print(f"{dataset_name}: ROUGE-L {rouge_l:.2f}")
        else:
            print(f"{dataset_name}: ROUGE-L unavailable")
    elif dataset_name in ("gsm8k", "arithmetic"):
        exact_match = 100.0 * correct / max(1, len(references))
        metrics["exact_match"] = exact_match
        metrics["correct"] = correct
        print(f"{dataset_name}: Exact Match {exact_match:.2f}% ({correct}/{len(references)})")

    metrics["predictions_file"] = str(pred_path)
    print(f"Wrote predictions to {pred_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to full_*.pt from model_full2.py")
    parser.add_argument("--trained_dataset", choices=["samsum", "gsm8k"], default=None)
    parser.add_argument(
        "--eval_datasets",
        type=parse_dataset_list,
        default=None,
        help="Comma-separated eval datasets. Defaults from --trained_dataset.",
    )
    parser.add_argument("--split", type=str, default=None, help="Override eval split for all datasets.")

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--base_model_name", type=str, default=None)
    parser.add_argument("--load_mode", choices=["ddm", "causal_lm"], default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=None)

    parser.add_argument("--max_examples", type=int, default=0, help="0 means evaluate the full split.")
    parser.add_argument("--arithmetic_samples_per_subtask", type=int, default=700)
    parser.add_argument("--metric_diffusion_steps", type=int, default=64)
    parser.add_argument("--metric_max_new_tokens", type=int, default=128)
    parser.add_argument("--metric_logits_temp", type=float, default=0.95)
    parser.add_argument("--metric_topp_temp", type=float, default=0.9)
    parser.add_argument("--shift", action="store_true", default=None)
    parser.add_argument("--no_shift", action="store_false", dest="shift")

    parser.add_argument("--output_dir", type=str, default="evaluation/full2_results")
    parser.add_argument("--output_prefix", type=str, default=None)
    parser.add_argument("--print_examples", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    state_dict, metadata = read_checkpoint(args.checkpoint)
    apply_metadata_defaults(args, metadata)
    if args.eval_datasets is None:
        args.eval_datasets = default_eval_datasets(args.trained_dataset)
    if args.output_prefix is None:
        args.output_prefix = Path(args.checkpoint).stem

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"Device: {device} | trained_dataset={args.trained_dataset} | "
        f"eval_datasets={','.join(args.eval_datasets)}"
    )
    print(
        "Decoding: "
        f"steps={args.metric_diffusion_steps}, max_new_tokens={args.metric_max_new_tokens}, "
        f"logits_temp={args.metric_logits_temp}, top_p={args.metric_topp_temp}, "
        f"max_len={args.max_len}, shift={args.shift}"
    )

    model, tokenizer = build_model(args, state_dict, device)
    del state_dict

    all_metrics = {
        "checkpoint": args.checkpoint,
        "trained_dataset": args.trained_dataset,
        "metadata": metadata,
        "decoding": {
            "metric_diffusion_steps": args.metric_diffusion_steps,
            "metric_max_new_tokens": args.metric_max_new_tokens,
            "metric_logits_temp": args.metric_logits_temp,
            "metric_topp_temp": args.metric_topp_temp,
            "max_len": args.max_len,
            "shift": args.shift,
        },
        "results": {},
    }

    for dataset_name in args.eval_datasets:
        if dataset_name not in ("samsum", "gsm8k", "arithmetic"):
            raise ValueError(f"Unsupported eval dataset: {dataset_name}")
        all_metrics["results"][dataset_name] = evaluate_dataset(
            model,
            tokenizer,
            args,
            dataset_name=dataset_name,
            device=device,
        )

    metrics_path = Path(args.output_dir) / f"{args.output_prefix}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=True)

    print(f"\nWrote metrics to {metrics_path}")
    print(json.dumps(all_metrics["results"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
