# Evaluation of baseline model. Trims the generated sequences at EOS tokens.
# EXAMPLE USAGE: 
# python eval_samsum_diffugpt.py --split=test --max_new_tokens=256
# Takes ~50 minutes on V100. 

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import DiscreteDiffusionModel, generate_samples


def build_prompt(dialogue: str) -> str:
    return f"Summarize the following dialogue.\n\nDialogue:\n{dialogue}\n\nSummary:"


def load_model_and_tokenizer(model_name: str, base_model_name: str, device: str):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DiscreteDiffusionModel.from_pretrained(
        model_name,
        model=base_model_name,
        config=config,
        tokenizer=tokenizer,
        device=device,
    ).to(device)
    model.eval()
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer has no mask token. Expected a DiffuGPT tokenizer.")
    return model, tokenizer


def get_target_token_count(args, tokenizer, reference_summary: str) -> int:
    if args.target_length_strategy == "gold":
        target_ids = tokenizer.encode(reference_summary, add_special_tokens=False)
        return max(1, min(len(target_ids), args.max_new_tokens))
    return args.max_new_tokens


def truncate_at_eos(token_ids, tokenizer):
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return token_ids
    if eos_token_id in token_ids:
        return token_ids[: token_ids.index(eos_token_id)]
    return token_ids


def generate_summary(model, tokenizer, args, dialogue: str, reference_summary: str) -> str:
    prompt = build_prompt(dialogue)
    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_source_tokens,
    )
    target_token_count = get_target_token_count(args, tokenizer, reference_summary)
    x0 = prompt_ids + [0] * target_token_count
    src_mask = [1] * len(prompt_ids) + [0] * target_token_count
    inputs = {
        "input_ids": torch.tensor([x0]),
        "src_mask": torch.tensor([src_mask]),
    }
    decode_args = args
    decode_args.diffusion_steps = args.diffusion_steps
    result = generate_samples(model, decode_args, tokenizer, inputs, verbose=args.verbose)
    pred_ids = result.tolist()[0][-target_token_count:]
    pred_ids = truncate_at_eos(pred_ids, tokenizer)
    return tokenizer.decode(pred_ids, skip_special_tokens=True).strip()


def evaluate_predictions(predictions, references, bertscore_model_type: str):
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")

    rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
    bertscore_scores = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type=bertscore_model_type,
    )

    return {
        "rougeL": round(rouge_scores["rougeL"] * 100, 4),
        "bertscore_precision": round(float(np.mean(bertscore_scores["precision"])) * 100, 4),
        "bertscore_recall": round(float(np.mean(bertscore_scores["recall"])) * 100, 4),
        "bertscore_f1": round(float(np.mean(bertscore_scores["f1"])) * 100, 4),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="diffusionfamily/diffugpt-m")
    parser.add_argument("--base_model_name", type=str, default="gpt2-medium")
    parser.add_argument("--dataset_name", type=str, default="knkarthick/samsum")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--max_source_tokens", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--target_length_strategy", type=str, choices=["fixed", "gold"], default="fixed")
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--logits_temp", type=float, default=0.95)
    parser.add_argument("--topp_temp", type=float, default=0.9)
    parser.add_argument("--shift", type=bool, default=True)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--bertscore_model_type", type=str, default="microsoft/deberta-xlarge-mnli")
    parser.add_argument("--output_file", type=str, default="evaluation/samsum_diffugpt_m_predictions.jsonl")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.base_model_name, args.device)
    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.max_examples > 0:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    predictions = []
    references = []

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(tqdm(dataset, total=len(dataset), desc="Evaluating")):
            prediction = generate_summary(model, tokenizer, args, sample["dialogue"], sample["summary"])
            reference = sample["summary"].strip()

            predictions.append(prediction)
            references.append(reference)
            row = {
                "index": idx,
                "dialogue": sample["dialogue"],
                "reference": reference,
                "prediction": prediction,
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            f.flush()

    print(f"Wrote predictions to {output_path}")

    metrics = evaluate_predictions(predictions, references, args.bertscore_model_type)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
