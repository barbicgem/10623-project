import re
import io
import csv
import contextlib
from argparse import ArgumentParser

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from model import DiscreteDiffusionModel, generate_samples
from model_lora2 import build_lora_diffugpt, load_lora


def extract_final_number(text):
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None

# def extract_final_number(text):
#     final_match = re.search(r"Final answer:\s*(-?\d+(?:\.\d+)?)", text)
#     if final_match:
#         return final_match.group(1)

#     matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
#     return matches[-1] if matches else None


def extract_gsm8k_gold(answer_text):
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "").rstrip(".")
    return extract_final_number(answer_text)


def build_model(args, config, tokenizer, device):
    # Match the newer training code default behavior
    model = DiscreteDiffusionModel.from_pretrained(
        args.model_name,
        model=args.base_model_name,
        config=config,
        tokenizer=tokenizer,
        device=device,
    )

    if args.use_lora:
        if args.lora_alpha is None:
            args.lora_alpha = 2 * args.rank

        model = build_lora_diffugpt(
            model,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_layers=None,
            train_embeddings=args.train_embeddings,
        )

        if args.lora_path is None:
            raise ValueError("--use_lora was set but --lora_path was not provided.")

        missing, unexpected = load_lora(model, args.lora_path, map_location="cpu")
        print(f"Loaded LoRA from {args.lora_path}")
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=str, default="diffusionfamily/diffugpt-m")
    parser.add_argument("--base_model_name", type=str, default="gpt2-medium")

    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--logits_temp", type=float, default=0.95)
    parser.add_argument("--topp_temp", type=float, default=0.9)
    parser.add_argument("--shift", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)

    # LoRA options
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--no_train_embeddings", action="store_false", dest="train_embeddings")
    parser.set_defaults(train_embeddings=True)
    

    parser.add_argument("--output_csv", type=str, default="gsm8k_results.csv")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = build_model(args, config, tokenizer, device)

    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"]

    gen_len = config.task_specific_params["text-generation"]["max_length"]

    correct = 0
    total = min(args.num_examples, len(test_data))

    with open(args.output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["prompt", "prediction", "gold", "correct", "raw_output"])

        for i in range(total):
            print(f"Running example {i + 1}/{total}", flush=True)

            ex = test_data[args.start_idx + i]
            question = ex["question"]
            gold = ex["answer"]

            # Match the newer model_lora2 prompt style
            prompt = f"Question: {question}\nAnswer:"

            prefix = [tokenizer.bos_token_id] + tokenizer.encode(prompt)

            if len(prefix) >= gen_len:
                prefix = prefix[:gen_len - 1]

            src_mask = [1] * len(prefix) + [0] * (gen_len - len(prefix))
            x0 = prefix + [0] * (gen_len - len(prefix))

            inputs = {
                "input_ids": torch.tensor([x0], device=device),
                "src_mask": torch.tensor([src_mask], device=device),
            }

            with torch.no_grad():
                with contextlib.redirect_stdout(io.StringIO()):
                    res = generate_samples(model, args, tokenizer, inputs, verbose=args.verbose)

            output_ids = res.tolist()[0]
            gen_ids = output_ids[len(prefix):]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            pred_num = extract_final_number(pred)
            gold_num = extract_gsm8k_gold(gold)
            is_correct = pred_num == gold_num

            if is_correct:
                correct += 1

            writer.writerow([prompt, pred_num, gold_num, is_correct, pred])

            print(
                f"[{i+1}/{total}] PRED={pred_num} GOLD={gold_num} CORRECT={is_correct}",
                flush=True,
            )
            print(f"RAW_PRED: {repr(pred)}", flush=True)

    print("=" * 80)
    print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")
    print(f"Saved results to {args.output_csv}")