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
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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


def build_model(model_name, rank, lora_alpha, checkpoint_path, device):
    config = AutoConfig.from_pretrained(model_name)
    backbone = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    backbone.config.use_cache = False
    ddm = DiscreteDiffusionModel(backbone, config, tokenizer, device=device)
    ddm = build_lora_diffugpt(ddm, r=rank, lora_alpha=lora_alpha)

    missing, unexpected = load_lora(ddm, checkpoint_path)
    if missing:
        print(f"Warning: {len(missing)} missing keys when loading LoRA checkpoint")
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
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, default=None, help="Defaults to 2 * rank")
    parser.add_argument("--dataset", choices=["samsum", "gsm8k", "arithmetic", "eleuther_arithmetic"], default="samsum")
    parser.add_argument("--model_name", type=str, default="diffusionfamily/diffugpt-m")
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--metric_diffusion_steps", type=int, default=64)
    parser.add_argument("--metric_max_new_tokens", type=int, default=64)
    parser.add_argument("--metric_logits_temp", type=float, default=0.95)
    parser.add_argument("--metric_topp_temp", type=float, default=0.9)
    parser.add_argument("--shift", action="store_true", default=True)
    parser.add_argument("--no_shift", action="store_false", dest="shift")
    args = parser.parse_args()

    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.rank

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | dataset={args.dataset} | rank={args.rank}")

    ddm, tokenizer = build_model(
        args.model_name, args.rank, args.lora_alpha, args.checkpoint, device
    )

    evaluate(ddm, tokenizer, args, device)
