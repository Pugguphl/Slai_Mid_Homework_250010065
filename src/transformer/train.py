"""Training entrypoint for the Transformer NMT baseline."""
from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path
from typing import Dict

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from sacrebleu import corpus_bleu
from torch.utils.data import DataLoader, DistributedSampler

from src.common.dataset import TranslationDataset, collate_translation_batch
from src.common.tokenizer import load_tokenizer
from src.common.utils import set_seed
from src.transformer.model import TransformerConfig, TransformerSeq2Seq


class NoamScheduler:
    def __init__(self, optimizer: optim.Optimizer, d_model: int, warmup_steps: int, factor: float = 1.0) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = max(1, warmup_steps)
        self.factor = factor
        self._step = 0

    def step(self) -> float:
        self._step += 1
        lr = self.factor * (self.d_model ** -0.5) * min(self._step ** -0.5, self._step * (self.warmup_steps ** -1.5))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloaders(config: Dict, tokenizer, distributed: bool, world_size: int, rank: int) -> Dict[str, DataLoader]:
    train_dataset = TranslationDataset(
        file_path=config["data"]["train_file"],
        tokenizer=tokenizer,
        src_key=config["data"].get("src_key", "zh"),
        tgt_key=config["data"].get("tgt_key", "en"),
        add_bos_eos=True,
    )
    valid_dataset = TranslationDataset(
        file_path=config["data"]["valid_file"],
        tokenizer=tokenizer,
        src_key=config["data"].get("src_key", "zh"),
        tgt_key=config["data"].get("tgt_key", "en"),
        add_bos_eos=True,
    )

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    collate_fn = partial(collate_translation_batch, tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 64),
        shuffle=train_sampler is None,
        num_workers=config["training"].get("num_workers", 2),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["training"].get("eval_batch_size", 64),
        shuffle=False,
        num_workers=config["training"].get("num_workers", 2),
        collate_fn=collate_fn,
    )
    return {"train": train_loader, "valid": valid_loader, "train_sampler": train_sampler}


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: NoamScheduler | None,
    criterion,
    device: torch.device,
    max_grad_norm: float,
    grad_accum_steps: int,
    distributed: bool,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    step = 0
    lr = scheduler.optimizer.param_groups[0]["lr"] if scheduler else optimizer.param_groups[0]["lr"]

    for batch_idx, batch in enumerate(dataloader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["src"], batch["src_lengths"], batch["tgt"])
        targets = batch["tgt"][:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss = loss / grad_accum_steps
        loss.backward()
        total_loss += loss.item() * grad_accum_steps

        if batch_idx % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                lr = scheduler.step()
            optimizer.zero_grad()
            step += 1

    remainder = len(dataloader) % grad_accum_steps
    if remainder != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler:
            lr = scheduler.step()
        optimizer.zero_grad()
        step += 1

    avg_loss = total_loss / max(step, 1)
    if distributed:
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / dist.get_world_size()
    return avg_loss, lr


def evaluate(model: TransformerSeq2Seq, dataloader: DataLoader, criterion, tokenizer, device: torch.device, max_decode_len: int) -> Dict:
    model.eval()
    total_loss = 0.0
    hyps, refs = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["src"], batch["src_lengths"], batch["tgt"])
            targets = batch["tgt"][:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()

            predictions = model.greedy_decode(batch["src"], batch["src_lengths"], max_len=max_decode_len)
            for pred_ids, tgt_ids in zip(predictions, batch["tgt"]):
                hyps.append(tokenizer.decode_tgt(pred_ids.tolist()))
                refs.append(tokenizer.decode_tgt(tgt_ids.tolist()[1:]))

    bleu = corpus_bleu(hyps, [refs]).score if hyps else 0.0
    return {"loss": total_loss / len(dataloader), "bleu": bleu}


def save_checkpoint(state: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer NMT model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--distributed", action="store_true", help="Force distributed mode")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = args.distributed or world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA GPUs")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0

    tokenizer_file = config["data"].get("tokenizer_file") or config["data"].get("vocab_model")
    if not tokenizer_file:
        raise KeyError("Config must provide data.tokenizer_file (or legacy/alt key data.vocab_model).")
    tokenizer = load_tokenizer(tokenizer_file)
    dataloaders = build_dataloaders(config, tokenizer, distributed, world_size, rank)

    model_cfg = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config["model"].get("d_model", 512),
        num_heads=config["model"].get("num_heads", 8),
        num_encoder_layers=config["model"].get("num_encoder_layers", 6),
        num_decoder_layers=config["model"].get("num_decoder_layers", 6),
        dim_feedforward=config["model"].get("dim_feedforward", 2048),
        dropout=config["model"].get("dropout", 0.1),
        activation=config["model"].get("activation", "relu"),
        max_position_embeddings=config["model"].get("max_position_embeddings", 256),
        max_decode_len=config["model"].get("max_decode_len", config["model"].get("max_position_embeddings", 256)),
        pad_idx=tokenizer.pad_id,
        bos_idx=tokenizer.bos_id,
        eos_idx=tokenizer.eos_id,
        share_embeddings=config["model"].get("share_embeddings", True),
        tie_output=config["model"].get("tie_output", True),
        norm_first=config["model"].get("norm_first", True),
        learned_positional_encoding=config["model"].get("learned_positional_encoding", False),
        # New parameters for ablation experiments
        positional_encoding=config["model"].get("positional_encoding", "sinusoidal"),
        norm_type=config["model"].get("norm_type", "layernorm"),
    )
    model = TransformerSeq2Seq(model_cfg).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id,
        label_smoothing=config["training"].get("label_smoothing", 0.0),
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"].get("learning_rate", 5e-4),
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    scheduler = None
    if config["training"].get("warmup_steps"):
        scheduler = NoamScheduler(optimizer, model_cfg.d_model, config["training"]["warmup_steps"], factor=1.0)

    grad_accum = config["training"].get("gradient_accumulation_steps", 1)
    max_grad_norm = config["training"].get("max_grad_norm", 1.0)
    best_bleu = 0.0
    history = []

    for epoch in range(1, config["training"].get("num_epochs", 10) + 1):
        sampler = dataloaders.get("train_sampler")
        if distributed and sampler is not None:
            sampler.set_epoch(epoch)
        start = time.time()
        train_loss, lr = train_epoch(
            model,
            dataloaders["train"],
            optimizer,
            scheduler,
            criterion,
            device,
            max_grad_norm,
            grad_accum,
            distributed,
        )
        eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        metrics = evaluate(
            eval_model,
            dataloaders["valid"],
            criterion,
            tokenizer,
            device,
            config["model"].get("max_decode_len", config["model"].get("max_position_embeddings", 256)),
        ) if rank == 0 else {"loss": 0.0, "bleu": 0.0}
        duration = time.time() - start
        if rank == 0:
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": metrics["loss"],
                "bleu": metrics["bleu"],
                "lr": lr,
                "time_sec": duration,
            })
            print(json.dumps(history[-1], ensure_ascii=False))

            if metrics["bleu"] > best_bleu:
                best_bleu = metrics["bleu"]
                save_path = Path(config["evaluation"].get("model_save_path", "experiments/logs/best_transformer.pt"))
                save_checkpoint({
                    "model_state": eval_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config,
                    "tokenizer": tokenizer_file,
                    "epoch": epoch,
                    "bleu": best_bleu,
                }, save_path)

    if rank == 0:
        metrics_path = Path(config.get("experiments", {}).get("metrics_file", "experiments/results/transformer_metrics.csv"))
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        pd.DataFrame(history).to_csv(metrics_path, index=False)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
