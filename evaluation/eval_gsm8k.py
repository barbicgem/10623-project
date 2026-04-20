import re
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from argparse import ArgumentParser
import contextlib
import io
import csv

from model import DiscreteDiffusionModel, generate_samples


def extract_final_number(text):
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None

def extract_gsm8k_gold(answer_text):
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "").rstrip(".")
    return extract_final_number(answer_text)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="diffusionfamily/diffugpt-m")
    parser.add_argument("--base_model_name", type=str, default="gpt2-medium")
    parser.add_argument("--shift", type=bool, default=True)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--logits_temp", type=float, default=0.95)
    parser.add_argument("--topp_temp", type=float, default=0.9)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--num_examples", type=int, default=10)
    args = parser.parse_args()

    model_name = args.model_name
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = DiscreteDiffusionModel.from_pretrained(
        model_name,
        model=args.base_model_name,
        config=config,
        tokenizer=tokenizer,
        device="cuda"
    ).to("cuda")

    model.eval()

    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"]

    gen_len = config.task_specific_params["text-generation"]["max_length"]

    correct = 0
    total = min(args.num_examples, len(test_data))
    csv_file = open("gsm8k_results.csv", "w", newline="")
    writer = csv.writer(csv_file)

    # header
    writer.writerow(["prompt", "prediction", "gold", "correct", "raw_output"])

    for i in range(total):
        print(f"Running example {i+1}/{total}", flush = True)

        ex = test_data[i]
        question = ex["question"]
        gold = ex["answer"]

        # Prompt format 1 -- working the best
        prompt = f"Question: {question}\nAnswer:"

        # # Prompt format 2
        # prompt = f"{question}\n"

        # # Prompt format 3 
        #prompt = f"Problem: {question}\nSolution:"

        prefix = [tokenizer.bos_token_id] + tokenizer.encode(prompt)

        if len(prefix) >= gen_len:
            prefix = prefix[:gen_len - 1]

        src_mask = [1] * len(prefix) + [0] * (gen_len - len(prefix))
        x0 = prefix + [0] * (gen_len - len(prefix))

        inputs = {
            "input_ids": torch.tensor([x0]).to("cuda"),
            "src_mask": torch.tensor([src_mask]).to("cuda"),
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
        print(f"[{i+1}/{total}] PROMPT = {prompt} PRED={pred_num} GOLD={gold_num} CORRECT={is_correct}", flush=True)

    csv_file.close()
    print("=" * 80)
    print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")