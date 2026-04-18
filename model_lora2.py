import math
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers.pytorch_utils import Conv1D
from typing import Optional

class LoRALayer(nn.Module):
    def __init__(self, base_layer: nn.Module, r: int, lora_alpha: int, dropout: float = 0.0):
        super().__init__()

        if r <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {r}")

        self.base_layer = base_layer
        self.r = r
        self.scale = lora_alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if isinstance(base_layer, nn.Linear):
            in_features = base_layer.in_features
            out_features = base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = base_layer.weight.shape
        else:
            raise TypeError(f"Unsupported layer type: {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scale * lora_out


def _get_parent_and_child(model: nn.Module, full_name: str):
    parts = full_name.split(".")
    if len(parts) == 1:
        return model, parts[0]
    parent_name = ".".join(parts[:-1])
    child_name = parts[-1]
    parent = model.get_submodule(parent_name)
    return parent, child_name


def inject_lora_attention_only(
    model: nn.Module,
    r: int,
    lora_alpha: int,
    dropout: float = 0.0,
    target_layers=None,
    verbose: bool = True,
):
    """
    Targets only GPT-style attention projections:
      - *.attn.c_attn
      - *.attn.c_proj

    This avoids accidentally hitting mlp.c_proj.
    """
    if target_layers is not None:
        target_layers = set(target_layers)

    replaced = []

    for name, module in list(model.named_modules()):
        leaf = name.split(".")[-1]
        parts = name.split(".")
        numeric_parts = [int(p) for p in parts if p.isdigit()]
        block_idx = numeric_parts[0] if numeric_parts else None

        is_target = (
            ("attn" in parts)
            and (leaf in {"c_attn", "c_proj"})
            and isinstance(module, (nn.Linear, Conv1D))
            and (target_layers is None or block_idx in target_layers)
        )

        if not is_target:
            continue

        parent, child = _get_parent_and_child(model, name)
        setattr(parent, child, LoRALayer(module, r=r, lora_alpha=lora_alpha, dropout=dropout))
        replaced.append(name)

    if verbose:
        print(f"Injected LoRA into {len(replaced)} attention modules")
        for n in replaced:
            print("  -", n)

    if len(replaced) == 0:
        print("Warning: no attention modules were LoRA-wrapped. Check module names.")

    return model


def mark_only_lora_as_trainable(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "embed_tokens" in name:
            p.requires_grad = True

    return model


def get_lora_state_dict(model: nn.Module):
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if ("lora_A" in name or "lora_B" in name or "embed_tokens" in name)
    }


def save_lora(model: nn.Module, path: str, metadata: Optional[dict] = None):
    payload = {
        "state_dict": get_lora_state_dict(model),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_lora(model: nn.Module, path: str, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    state_dict = payload["state_dict"] if "state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected


def build_lora_diffugpt(base_ddm, r=8, lora_alpha=16, dropout=0.05, target_layers=None):
    inject_lora_attention_only(
        base_ddm,
        r=r,
        lora_alpha=lora_alpha,
        dropout=dropout,
        target_layers=target_layers,
        verbose=True,
    )
    mark_only_lora_as_trainable(base_ddm)
    return base_ddm


def make_gsm8k_dataloader(tokenizer, split="train", max_len=256, batch_size=16):
    ds = load_dataset("openai/gsm8k", "main", split=split)

    def tokenize(example):
        text = example["question"] + " " + example["answer"]
        enc = tokenizer(text, truncation=True, max_length=max_len,
                        padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0)}

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ds.set_format("torch")
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))


def make_samsum_dataloader(tokenizer, split="train", max_len=256, batch_size=16):
    ds = load_dataset("knkarthick/samsum", split=split)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def tokenize(example):
        prompt = example["dialogue"] + " TL;DR: "
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(example["summary"], add_special_tokens=False)

        # leave room for at least a few summary tokens
        prompt_ids = prompt_ids[:max_len - 8]
        max_target_len = max_len - len(prompt_ids)
        target_ids = target_ids[:max_target_len]

        q_len = len(prompt_ids)
        t_len = len(target_ids)

        full_ids = prompt_ids + target_ids
        pad_len = max_len - len(full_ids)
        full_ids = full_ids + [pad_id] * pad_len

        src_mask = [True] * q_len + [False] * (t_len + pad_len)
        answer_mask = [False] * q_len + [True] * t_len + [False] * pad_len

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "src_mask": torch.tensor(src_mask, dtype=torch.bool),
            "answer_mask": torch.tensor(answer_mask, dtype=torch.bool),
        }

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ds.set_format("torch")
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))


def transition(x_0, sigma, maskable_mask, mask_token_id):
    move_indices = (torch.rand(*x_0.shape, device=x_0.device) < sigma) & maskable_mask
    return torch.where(move_indices, mask_token_id, x_0)


def get_bidirectional_attn_mask(seq_len, bsz, dtype, device):
    visible = torch.ones((bsz, 1, seq_len, seq_len), dtype=dtype, device=device)
    inverted = 1.0 - visible
    return inverted.masked_fill(inverted.to(torch.bool), torch.finfo(dtype).min)


def diffusion_loss(model, batch, mask_token_id, shift=True, sampling_eps=1e-3):
    x = batch["input_ids"]
    answer_mask = batch["answer_mask"]   # required; excludes prompt and padding

    B, L = x.shape

    t = (1.0 - sampling_eps) * torch.rand(B, device=x.device) + sampling_eps
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
        L, B, dtype=x_embed.dtype, device=x.device
    )
    logits = model(x_t, attention_mask=attention_mask)

    # Only train on masked target tokens
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
    ).reshape(B, -1)

    token_loss = token_loss.masked_fill(~loss_mask, 0.0)
    final_loss = (dsigma[:, None] * token_loss).sum() / loss_mask.sum().clamp_min(1)

    return final_loss

def eval_loss(model, val_loader, mask_token_id, pad_token_id, device, num_batches=20):
    model.eval()
    total_loss, count = 0.0, 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total_loss += diffusion_loss(model, batch, mask_token_id, shift=True).item()
            count += 1
            if count >= num_batches:
                break

    model.train()
    return total_loss / max(count, 1)


def train(
    model,
    tokenizer,
    rank,
    mask_token_id,
    pad_token_id,
    num_steps=3000,
    lr=1e-4,
    warmup_steps=100,
    batch_size=16,
    max_len=256,
    device="cpu",
    grad_clip=1.0,
):
    train_loader = make_samsum_dataloader(
        tokenizer, split="train", max_len=max_len, batch_size=batch_size
    )

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_steps - warmup_steps))

    model.to(device)
    model.train()

    step = 0
    for batch in cycle(train_loader):
        if step >= num_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        loss = diffusion_loss(model, batch, mask_token_id, shift=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=grad_clip,
        )

        optimizer.step()

        if step < warmup_steps:
            warmup_lr = lr * (step + 1) / warmup_steps
            for g in optimizer.param_groups:
                g["lr"] = warmup_lr
        else:
            scheduler.step()

        if step % 50 == 0:
            print(
                f"step {step} | train_loss {loss.item():.4f} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e}"
            )

        step += 1

    return model


def print_lora_summary(model):
    print("\nLoRA modules:")
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            print(f"  {name}: in={module.in_features}, out={module.out_features}")
            count += 1
    print(f"Total LoRA modules: {count}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")


if __name__ == "__main__":
    import argparse
    import os

    import transformers.models.llama.modeling_llama as _llama
    if not hasattr(_llama, "LlamaFlashAttention2"):
        _llama.LlamaFlashAttention2 = _llama.LlamaAttention

    from model import DiscreteDiffusionModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | rank={args.rank} | seed={args.seed}")

    model_name = "diffusionfamily/diffugpt-m"
    config = AutoConfig.from_pretrained(model_name)
    backbone = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for block in backbone.transformer.h:
        block.attn.register_buffer(
            "bias",
            torch.tril(
                torch.ones(config.n_positions, config.n_positions, dtype=torch.bool)
            ).view(1, 1, config.n_positions, config.n_positions),
        )

    backbone.config.use_cache = False
    ddm = DiscreteDiffusionModel(backbone, config, tokenizer, device=device)

    ddm = build_lora_diffugpt(
        ddm,
        r=args.rank,
        lora_alpha=2 * args.rank,
        dropout=0.05,
        target_layers=None,
    )

    print_lora_summary(ddm)

    mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 50257
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    ddm = train(
        ddm,
        tokenizer,
        rank=args.rank,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        num_steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        max_len=args.max_len,
        device=device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"lora_r{args.rank}_seed{args.seed}.pt")
    save_lora(
        ddm,
        save_path,
        metadata={
            "rank": args.rank,
            "lora_alpha": 2 * args.rank,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "model_name": model_name,
            "target": "attention_only",
        },
    )
    print(f"Saved to {save_path}")
