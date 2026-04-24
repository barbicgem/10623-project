"""
Full finetuning for DiffuGPT-medium with the DDM-SFT loss.

Quick start:
  python model_full2.py --dataset samsum --steps 6000 --batch_size 16 --lr 5e-5 --precision fp16
  python model_full2.py --dataset gsm8k --batch_size 8 --precision fp16

Most common knobs:
  --dataset {samsum,gsm8k}    task/data to finetune on
  --lr 1e-4                   learning rate
  --batch_size 16             per-step batch size
  --precision auto            auto, fp16, bf16, or fp32. Use fp16 for V100, bf16 for H100
  --log_every 10              training log frequency
  --eval_every 100            cheap denoising-loss eval frequency
  --metric_every 1000         expensive ROUGE-L/GSM exact-match generation eval frequency

Notes:
  - W&B is on by default; use --no_wandb to disable.
  - This script trains all DiffuGPT parameters, not LoRA adapters.
  - Metric eval logs a few prompt/reference/prediction examples to W&B.
  - Use --help for all advanced options.
"""

import argparse
import io
import math
import os
import re
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict, dataclass
from itertools import cycle
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader


def build_samsum_prompt(dialogue: str) -> str:
    return f"Summarize the following dialogue.\n\nDialogue:\n{dialogue}\n\nSummary:"


def build_gsm8k_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def make_prompt_target(example: dict, dataset_name: str):
    if dataset_name == "samsum":
        return build_samsum_prompt(example["dialogue"]), example["summary"]
    if dataset_name == "gsm8k":
        return build_gsm8k_prompt(example["question"]), example["answer"]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def encode_prompt_target(
    tokenizer,
    prompt: str,
    target: str,
    max_len: int,
    min_target_tokens: int,
    pad_id: int,
):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)

    if len(target_ids) == 0:
        target_ids = [tokenizer.eos_token_id or pad_id]

    min_target_tokens = max(1, min(min_target_tokens, max_len))
    max_prompt_len = max(0, max_len - min_target_tokens)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:] if max_prompt_len > 0 else []

    target_budget = max_len - len(prompt_ids)
    target_ids = target_ids[:target_budget]
    if len(target_ids) == 0:
        target_ids = [tokenizer.eos_token_id or pad_id]
        if len(prompt_ids) + len(target_ids) > max_len:
            prompt_ids = prompt_ids[: max_len - 1]

    q_len = len(prompt_ids)
    t_len = len(target_ids)
    full_ids = prompt_ids + target_ids
    pad_len = max_len - len(full_ids)

    input_ids = full_ids + [pad_id] * pad_len
    src_mask = [True] * q_len + [False] * (t_len + pad_len)
    answer_mask = [False] * q_len + [True] * t_len + [False] * pad_len

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "src_mask": torch.tensor(src_mask, dtype=torch.bool),
        "answer_mask": torch.tensor(answer_mask, dtype=torch.bool),
    }


def dataset_default_split(dataset_name: str, train: bool):
    if dataset_name == "samsum":
        return "train" if train else "validation"
    if dataset_name == "gsm8k":
        return "train" if train else "test"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_named_dataset(dataset_name: str, split: str):
    from datasets import load_dataset

    if dataset_name == "samsum":
        return load_dataset("knkarthick/samsum", split=split)
    if dataset_name == "gsm8k":
        return load_dataset("openai/gsm8k", "main", split=split)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def make_supervised_dataloader(
    tokenizer,
    dataset_name: str,
    split: str,
    max_len: int,
    batch_size: int,
    min_target_tokens: int,
    max_samples: int = 0,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    ds = load_named_dataset(dataset_name, split)
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    def tokenize(example):
        prompt, target = make_prompt_target(example, dataset_name)
        return encode_prompt_target(
            tokenizer,
            prompt=prompt,
            target=target,
            max_len=max_len,
            min_target_tokens=min_target_tokens,
            pad_id=pad_id,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ds.set_format("torch")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def transition(x_0, sigma, maskable_mask, mask_token_id):
    move_indices = (torch.rand(*x_0.shape, device=x_0.device) < sigma) & maskable_mask
    return torch.where(move_indices, mask_token_id, x_0)


def get_bidirectional_attn_mask(seq_len, bsz, dtype, device):
    visible = torch.ones((bsz, 1, seq_len, seq_len), dtype=dtype, device=device)
    inverted = 1.0 - visible
    return inverted.masked_fill(inverted.to(torch.bool), torch.finfo(dtype).min)


def diffusion_loss(model, batch, mask_token_id, shift=True, sampling_eps=1e-3):
    x = batch["input_ids"]
    answer_mask = batch["answer_mask"]

    batch_size, seq_len = x.shape

    t = (1.0 - sampling_eps) * torch.rand(batch_size, device=x.device) + sampling_eps
    sigma = t
    dsigma = 1.0 / t

    x_t = transition(
        x,
        sigma[:, None],
        maskable_mask=answer_mask,
        mask_token_id=mask_token_id,
    )

    x_embed = model.get_embeds(x_t)
    attention_mask = get_bidirectional_attn_mask(
        seq_len, batch_size, dtype=x_embed.dtype, device=x.device
    )
    logits = model(x_t, attention_mask=attention_mask)

    loss_mask = (x_t == mask_token_id) & answer_mask

    if shift:
        logits = logits[:, :-1, :]
        loss_mask = loss_mask[:, 1:]
        labels = x[:, 1:]
    else:
        labels = x

    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).reshape(batch_size, -1)

    token_loss = token_loss.masked_fill(~loss_mask, 0.0)
    masked_tokens = loss_mask.sum().clamp_min(1)
    final_loss = (dsigma[:, None] * token_loss).sum() / masked_tokens
    unweighted_loss = token_loss.sum() / masked_tokens

    return final_loss, unweighted_loss, masked_tokens.detach()


@dataclass
class TrainConfig:
    dataset: str
    train_split: str
    eval_split: str
    output_dir: str
    steps: int
    batch_size: int
    max_len: int
    min_target_tokens: int
    lr: float
    min_lr_ratio: float
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    precision: str
    shift: bool
    sampling_eps: float
    metric_every: int
    metric_max_examples: int
    metric_diffusion_steps: int
    metric_max_new_tokens: int


def compute_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_lr_ratio: float):
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps

    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def set_optimizer_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


def make_optimizer(model, lr: float, weight_decay: float):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "embed_tokens" not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def resolve_precision(precision: str, device: torch.device):
    if precision == "auto":
        if device.type != "cuda":
            return "fp32"
        major, _ = torch.cuda.get_device_capability(device)
        return "bf16" if major >= 8 else "fp16"
    return precision


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def resolve_run_name(args):
    if args.run_name is not None:
        return args.run_name
    return f"{args.dataset}-full-lr{args.lr:g}-bs{args.batch_size}"


def sanitize_path_component(name: str):
    sanitized = re.sub(r"[\\/]+", "-", name.strip())
    sanitized = re.sub(r"\s+", "-", sanitized)
    return sanitized or "run"


def configure_output_dir(args):
    run_name = resolve_run_name(args)
    args.run_name = run_name
    args.output_dir = os.path.join(args.output_dir, sanitize_path_component(run_name))
    os.makedirs(args.output_dir, exist_ok=True)
    return run_name


def init_wandb(args, config: TrainConfig, trainable_params: int, total_params: int):
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError:
        print("wandb is enabled but not installed; continuing without W&B logging.")
        return None

    return wandb.init(
        entity="yuzhenzh-carnegie-mellon-university",
        project=args.wandb_project,
        name=args.run_name,
        mode=args.wandb_mode,
        config={
            **asdict(config),
            "model_name": args.model_name,
            "base_model_name": args.base_model_name,
            "load_mode": args.load_mode,
            "finetune_mode": "full",
            "trainable_params": trainable_params,
            "total_params": total_params,
        },
    )


def log_metrics(wandb_run, metrics: dict, step: int):
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def eval_loss(
    model,
    val_loader,
    mask_token_id,
    device,
    precision: str,
    shift: bool,
    sampling_eps: float,
    num_batches: int = 20,
):
    model.eval()
    total_loss, total_unweighted, total_tokens, count = 0.0, 0.0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_context(device, precision):
                loss, unweighted_loss, masked_tokens = diffusion_loss(
                    model,
                    batch,
                    mask_token_id=mask_token_id,
                    shift=shift,
                    sampling_eps=sampling_eps,
                )
            total_loss += loss.item()
            total_unweighted += unweighted_loss.item()
            total_tokens += int(masked_tokens.item())
            count += 1
            if count >= num_batches:
                break

    model.train()
    return {
        "eval/loss": total_loss / max(count, 1),
        "eval/unweighted_loss": total_unweighted / max(count, 1),
        "eval/masked_tokens": total_tokens / max(count, 1),
    }


def load_metric_examples(dataset_name: str, split: str, max_examples: int):
    if max_examples <= 0:
        return []

    ds = load_named_dataset(dataset_name, split)
    ds = ds.select(range(min(max_examples, len(ds))))
    examples = []
    for example in ds:
        prompt, target = make_prompt_target(example, dataset_name)
        examples.append({"prompt": prompt, "target": target})
    return examples


def truncate_at_eos(token_ids, tokenizer):
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None or eos_token_id not in token_ids:
        return token_ids
    return token_ids[: token_ids.index(eos_token_id)]


def generate_target_text(model, tokenizer, args, prompt: str, target: str, device):
    from model import generate_samples

    target_ids = tokenizer.encode(target, add_special_tokens=False)
    target_token_count = max(1, min(len(target_ids), args.metric_max_new_tokens, args.max_len))
    max_prompt_len = max(0, args.max_len - target_token_count)
    if max_prompt_len > 0:
        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_len,
        )
    else:
        prompt_ids = []

    x0 = prompt_ids + [0] * target_token_count
    src_mask = [1] * len(prompt_ids) + [0] * target_token_count
    inputs = {
        "input_ids": torch.tensor([x0], device=device),
        "src_mask": torch.tensor([src_mask], device=device),
    }
    decode_args = SimpleNamespace(
        diffusion_steps=args.metric_diffusion_steps,
        logits_temp=args.metric_logits_temp,
        topp_temp=args.metric_topp_temp,
        shift=args.shift,
    )

    with redirect_stdout(io.StringIO()):
        result = generate_samples(model, decode_args, tokenizer, inputs, verbose=False)
    pred_ids = result.tolist()[0][-target_token_count:]
    pred_ids = truncate_at_eos(pred_ids, tokenizer)
    return tokenizer.decode(pred_ids, skip_special_tokens=True).strip()


def extract_gsm_answer(text: str):
    after_marker = re.search(r"####\s*([-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?)", text)
    if after_marker:
        return after_marker.group(1).replace(",", "")

    numbers = re.findall(r"[-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?", text)
    if not numbers:
        return ""
    return numbers[-1].replace(",", "")


def compute_rouge_l(predictions, references):
    try:
        import evaluate
    except ImportError:
        print("evaluate is not installed; skipping ROUGE-L metric.")
        return None

    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=predictions, references=references)
    return float(scores["rougeL"] * 100)


def make_examples_table(wandb_run, rows):
    if wandb_run is None:
        return None

    try:
        import wandb
    except ImportError:
        return None

    table = wandb.Table(columns=["index", "prompt", "reference", "prediction"])
    for row in rows:
        table.add_data(row["index"], row["prompt"], row["reference"], row["prediction"])
    return table


def eval_generation_metrics(model, tokenizer, args, metric_examples, device, wandb_run=None):
    if not metric_examples:
        return {}

    model.eval()
    predictions = []
    references = []
    example_rows = []

    with torch.no_grad():
        for idx, example in enumerate(metric_examples):
            prediction = generate_target_text(
                model,
                tokenizer,
                args,
                prompt=example["prompt"],
                target=example["target"],
                device=device,
            )
            predictions.append(prediction)
            references.append(example["target"])

            if idx < args.metric_examples_to_log:
                example_rows.append(
                    {
                        "index": idx,
                        "prompt": example["prompt"],
                        "reference": example["target"],
                        "prediction": prediction,
                    }
                )

    metrics = {"metric/examples": len(metric_examples)}
    if args.dataset == "samsum":
        rouge_l = compute_rouge_l(predictions, references)
        if rouge_l is not None:
            metrics["metric/rougeL"] = rouge_l
    elif args.dataset == "gsm8k":
        correct = 0
        for pred, ref in zip(predictions, references):
            correct += int(extract_gsm_answer(pred) == extract_gsm_answer(ref))
        metrics["metric/exact_match"] = 100.0 * correct / max(1, len(references))

    table = make_examples_table(wandb_run, example_rows)
    if table is not None:
        metrics[f"examples/{args.dataset}"] = table

    model.train()
    return metrics


def save_full_checkpoint(model, args, config: TrainConfig, step: int, suffix: str):
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"full_{suffix}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": {
                **asdict(config),
                "step": step,
                "model_name": args.model_name,
                "base_model_name": args.base_model_name,
                "load_mode": args.load_mode,
                "target": "full_finetune",
            },
        },
        save_path,
    )
    print(f"Saved {save_path}")
    return save_path


def load_full_checkpoint(model, path: str, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    state_dict = payload["state_dict"] if "state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected


def train(
    model,
    train_loader,
    val_loader,
    metric_examples,
    tokenizer,
    mask_token_id,
    device,
    args,
    config: TrainConfig,
    wandb_run=None,
):
    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    precision = resolve_precision(args.precision, device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and precision == "fp16"))

    model.to(device)
    model.train()

    optimizer_step = 0
    running_loss = 0.0
    running_unweighted = 0.0
    running_masked_tokens = 0
    train_iter = cycle(train_loader)
    optimizer.zero_grad(set_to_none=True)

    while optimizer_step < args.steps:
        next_lr = compute_lr(
            optimizer_step + 1,
            total_steps=args.steps,
            base_lr=args.lr,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        set_optimizer_lr(optimizer, next_lr)

        batch = next(train_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast_context(device, precision):
            loss, unweighted_loss, masked_tokens = diffusion_loss(
                model,
                batch,
                mask_token_id=mask_token_id,
                shift=args.shift,
                sampling_eps=args.sampling_eps,
            )

        scaler.scale(loss).backward()
        running_loss += loss.item()
        running_unweighted += unweighted_loss.item()
        running_masked_tokens += int(masked_tokens.item())

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=args.grad_clip,
            )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1

        if optimizer_step % args.log_every == 0 or optimizer_step == 1:
            denom = args.log_every if optimizer_step % args.log_every == 0 else 1
            metrics = {
                "train/loss": running_loss / denom,
                "train/unweighted_loss": running_unweighted / denom,
                "train/lr": next_lr,
                "train/masked_tokens": running_masked_tokens / denom,
            }
            print(
                f"step {optimizer_step} | loss {metrics['train/loss']:.4f} | "
                f"unw {metrics['train/unweighted_loss']:.4f} | lr {next_lr:.2e}"
            )
            log_metrics(wandb_run, metrics, optimizer_step)
            running_loss = 0.0
            running_unweighted = 0.0
            running_masked_tokens = 0

        if val_loader is not None and args.eval_every > 0 and optimizer_step % args.eval_every == 0:
            metrics = eval_loss(
                model,
                val_loader,
                mask_token_id=mask_token_id,
                device=device,
                precision=precision,
                shift=args.shift,
                sampling_eps=args.sampling_eps,
                num_batches=args.eval_batches,
            )
            print(
                f"eval step {optimizer_step} | loss {metrics['eval/loss']:.4f} | "
                f"unw {metrics['eval/unweighted_loss']:.4f}"
            )
            log_metrics(wandb_run, metrics, optimizer_step)

        if metric_examples and args.metric_every > 0 and optimizer_step % args.metric_every == 0:
            metrics = eval_generation_metrics(
                model,
                tokenizer,
                args,
                metric_examples=metric_examples,
                device=device,
                wandb_run=wandb_run,
            )
            printable = {k: v for k, v in metrics.items() if not k.startswith("examples/")}
            print(f"metric eval step {optimizer_step} | {printable}")
            log_metrics(wandb_run, metrics, optimizer_step)

        if args.save_every > 0 and optimizer_step % args.save_every == 0:
            save_full_checkpoint(model, args, config, optimizer_step, suffix=f"step{optimizer_step}")

    return model


def print_trainable_summary(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    return trainable, total


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_diffugpt(args, tokenizer, config, device):
    from model import DiscreteDiffusionModel
    from transformers import AutoModelForCausalLM

    if args.load_mode == "ddm":
        return DiscreteDiffusionModel.from_pretrained(
            args.model_name,
            model=args.base_model_name,
            config=config,
            tokenizer=tokenizer,
            device=device,
        )

    backbone = AutoModelForCausalLM.from_pretrained(args.model_name)
    backbone.config.use_cache = False
    return DiscreteDiffusionModel(backbone, config, tokenizer, device=device)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Full finetuning for DiffuGPT-medium with the DDM-SFT loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model_name", type=str, default="diffusionfamily/diffugpt-m")
    parser.add_argument("--base_model_name", type=str, default="gpt2-medium")
    parser.add_argument("--load_mode", choices=["ddm", "causal_lm"], default="ddm")
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    parser.add_argument("--dataset", choices=["samsum", "gsm8k"], default="samsum")
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--min_target_tokens", type=int, default=16)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--sampling_eps", type=float, default=1e-3)
    parser.add_argument("--no_shift", action="store_false", dest="shift")
    parser.set_defaults(shift=True)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--metric_every", type=int, default=500)
    parser.add_argument("--metric_max_examples", type=int, default=64)
    parser.add_argument("--metric_examples_to_log", type=int, default=4)
    parser.add_argument("--metric_diffusion_steps", type=int, default=64)
    parser.add_argument("--metric_max_new_tokens", type=int, default=128)
    parser.add_argument("--metric_logits_temp", type=float, default=0.95)
    parser.add_argument("--metric_topp_temp", type=float, default=0.9)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="output/diffugpt-m-full")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", action="store_false", dest="wandb")
    parser.add_argument("--wandb_project", type=str, default="423-final-proj")
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--run_name", type=str, default=None)

    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.train_split is None:
        args.train_split = dataset_default_split(args.dataset, train=True)
    if args.eval_split is None:
        args.eval_split = dataset_default_split(args.dataset, train=False)

    run_name = configure_output_dir(args)
    set_seed(args.seed)

    import transformers.models.llama.modeling_llama as _llama
    if not hasattr(_llama, "LlamaFlashAttention2"):
        _llama.LlamaFlashAttention2 = _llama.LlamaAttention

    from transformers import AutoConfig, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = resolve_precision(args.precision, device)
    print(f"Device: {device} | precision={precision} | dataset={args.dataset}")
    print(f"Run: {run_name} | checkpoints: {args.output_dir}")

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer has no mask token. Expected a DiffuGPT tokenizer.")

    ddm = load_diffugpt(args, tokenizer=tokenizer, config=config, device=device)

    if args.resume_checkpoint:
        missing, unexpected = load_full_checkpoint(ddm, args.resume_checkpoint, map_location="cpu")
        print(
            f"Loaded full checkpoint from {args.resume_checkpoint}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    trainable_params, total_params = print_trainable_summary(ddm)

    pin_memory = device.type == "cuda"
    train_loader = make_supervised_dataloader(
        tokenizer,
        dataset_name=args.dataset,
        split=args.train_split,
        max_len=args.max_len,
        batch_size=args.batch_size,
        min_target_tokens=args.min_target_tokens,
        max_samples=args.max_train_samples,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = None
    if args.eval_every > 0 and args.eval_batches > 0:
        val_loader = make_supervised_dataloader(
            tokenizer,
            dataset_name=args.dataset,
            split=args.eval_split,
            max_len=args.max_len,
            batch_size=args.batch_size,
            min_target_tokens=args.min_target_tokens,
            max_samples=args.max_eval_samples,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    train_config = TrainConfig(
        dataset=args.dataset,
        train_split=args.train_split,
        eval_split=args.eval_split,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        max_len=args.max_len,
        min_target_tokens=args.min_target_tokens,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        precision=precision,
        shift=args.shift,
        sampling_eps=args.sampling_eps,
        metric_every=args.metric_every,
        metric_max_examples=args.metric_max_examples,
        metric_diffusion_steps=args.metric_diffusion_steps,
        metric_max_new_tokens=args.metric_max_new_tokens,
    )
    wandb_run = init_wandb(args, train_config, trainable_params, total_params)

    metric_examples = []
    if args.metric_every > 0 and args.metric_max_examples > 0:
        metric_examples = load_metric_examples(
            args.dataset,
            split=args.eval_split,
            max_examples=args.metric_max_examples,
        )

    ddm = train(
        ddm,
        train_loader=train_loader,
        val_loader=val_loader,
        metric_examples=metric_examples,
        tokenizer=tokenizer,
        mask_token_id=tokenizer.mask_token_id,
        device=device,
        args=args,
        config=train_config,
        wandb_run=wandb_run,
    )

    save_full_checkpoint(ddm, args, train_config, args.steps, suffix="final")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
