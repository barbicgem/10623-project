"""
Evaluate a trained LoRA DiffuGPT checkpoint on SAMSum or GSM8K.

Usage:
    python eval_lora.py --checkpoint output/lora_r8_seed0.pt --rank 8 --dataset samsum
    python eval_lora.py --checkpoint output/lora_r16_seed0.pt --rank 16 --dataset gsm8k

Metrics:
    samsum  -> ROUGE-L
    gsm8k   -> Exact match (numeric answer)
"""

import argparse
import json
import torch
from transformers import AutoConfig, AutoTokenizer

import transformers.models.llama.modeling_llama as _llama
if not hasattr(_llama, "LlamaFlashAttention2"):
    _llama.LlamaFlashAttention2 = _llama.LlamaAttention

from model import DiscreteDiffusionModel
from model_lora2 import (
    build_lora_diffugpt,
    load_lora,
    load_named_dataset,
    make_prompt_target,
    generate_target_text,
    compute_rouge_l,
    extract_gsm_answer,
)


def build_model(model_name, rank, lora_alpha, checkpoint_path, device, base_model_name="gpt2-medium", baseline=False):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ddm = DiscreteDiffusionModel.from_pretrained(
        model_name,
        model=base_model_name,
        config=config,
        tokenizer=tokenizer,
        device=device,
    )

    if not baseline:
        ddm = build_lora_diffugpt(ddm, r=rank, lora_alpha=lora_alpha)
        missing, unexpected = load_lora(ddm, checkpoint_path)
        lora_missing = [k for k in missing if "lora_A" in k or "lora_B" in k]
        if lora_missing:
            print(f"ERROR: {len(lora_missing)} LoRA keys missing from checkpoint — weights not loaded!")
            print("  First few:", lora_missing[:4])
        else:
            print(f"LoRA loaded OK ({len(missing)} backbone keys not in checkpoint, as expected)")
        if unexpected:
            print(f"WARNING: {len(unexpected)} unexpected keys in checkpoint (key mismatch — LoRA may not have loaded)")
            print("  First few:", unexpected[:4])
    else:
        print("Baseline mode — no LoRA, using base DiffuGPT weights only")

    ddm = ddm.to(device)
    ddm.eval()
    return ddm, tokenizer


def evaluate(ddm, tokenizer, args, device):
    split = "test"
    ds = load_named_dataset(args.dataset, split, max_samples_per_subtask=700)  # ~2.1k for arithmetic

    if args.max_examples and args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    predictions = []
    references = []

    for idx, example in enumerate(ds):
        prompt, target = make_prompt_target(example, args.dataset)

        prediction = generate_target_text(
            ddm,
            tokenizer,
            args,
            prompt=prompt,
            target=target,
            device=device,
        )

        predictions.append(prediction)
        references.append(target)

        if idx < 5:
            print(f"\n--- Example {idx} ---")
            print(f"Ref:  {target[:200]}")
            print(f"Pred: {prediction[:200]}")

        if (idx + 1) % 50 == 0:
            print(f"[{idx + 1}/{len(ds)}] processed...")

    print(f"\nResults ({len(predictions)} examples):")

    if args.dataset == "samsum":
        rouge_l = compute_rouge_l(predictions, references)
        if rouge_l is not None:
            print(f"  ROUGE-L: {rouge_l:.2f}")
        return {"rougeL": rouge_l}

    elif args.dataset in ("gsm8k", "arithmetic", "eleuther_arithmetic"):
        correct = sum(
            int(extract_gsm_answer(pred) == extract_gsm_answer(ref))
            for pred, ref in zip(predictions, references)
        )
        exact_match = 100.0 * correct / max(1, len(references))
        print(f"  Exact Match: {exact_match:.2f}%  ({correct}/{len(references)})")
        return {"exact_match": exact_match}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--baseline", action="store_true", default=False,
                        help="Evaluate base DiffuGPT with no LoRA (ignores --checkpoint and --rank)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None, help="Defaults to 2 * rank")
    parser.add_argument("--dataset", choices=["samsum", "gsm8k", "arithmetic", "eleuther_arithmetic"], default="samsum")
    parser.add_argument("--model_name", type=str, default="diffusionfamily/diffugpt-m")
    parser.add_argument("--base_model_name", type=str, default="gpt2-medium")
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--metric_diffusion_steps", type=int, default=64)
    parser.add_argument("--metric_max_new_tokens", type=int, default=64)
    parser.add_argument("--metric_logits_temp", type=float, default=0.95)
    parser.add_argument("--metric_topp_temp", type=float, default=0.9)
    parser.add_argument("--shift", action="store_true", default=True)
    parser.add_argument("--no_shift", action="store_false", dest="shift")
    parser.add_argument("--output_json", type=str, default=None, help="If set, save metrics dict as JSON to this path")
    args = parser.parse_args()

    if not args.baseline and args.checkpoint is None:
        parser.error("--checkpoint is required unless --baseline is set")
    if not args.baseline and args.rank is None:
        parser.error("--rank is required unless --baseline is set")

    if args.lora_alpha is None and args.rank is not None:
        args.lora_alpha = 2 * args.rank

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "baseline" if args.baseline else f"rank={args.rank}"
    print(f"Device: {device} | dataset={args.dataset} | {mode}")

    ddm, tokenizer = build_model(
        args.model_name, args.rank, args.lora_alpha, args.checkpoint, device,
        base_model_name=args.base_model_name,
        baseline=args.baseline,
    )

    metrics = evaluate(ddm, tokenizer, args, device)

    if args.output_json and metrics:
        payload = {"dataset": args.dataset, "rank": args.rank, "baseline": args.baseline, **metrics}
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics to {args.output_json}")
