"""Training entrypoint for the RNN NMT system."""
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
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import yaml
from sacrebleu import corpus_bleu
from torch.utils.data import DataLoader, DistributedSampler

from src.common.dataset import TranslationDataset, collate_translation_batch
from src.common.tokenizer import load_tokenizer
from src.common.utils import set_seed
from src.rnn.model import Seq2Seq, Seq2SeqConfig


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
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    collate_fn = partial(collate_translation_batch, tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 64),
        shuffle=train_sampler is None,
        num_workers=config["training"].get("num_workers", 2),
        collate_fn=collate_fn,
        sampler=train_sampler,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["training"].get("eval_batch_size", 32),
        shuffle=False,
        num_workers=config["training"].get("num_workers", 2),
        collate_fn=collate_fn,
    )
    return {"train": train_loader, "valid": valid_loader, "train_sampler": train_sampler}


def train_epoch(model: Seq2Seq, dataloader: DataLoader, optimizer, criterion, device, teacher_forcing: float, grad_clip: float, distributed: bool) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch["src"], batch["src_lengths"], batch["tgt"], teacher_forcing)
        logits = outputs[:, 1:, :].reshape(-1, outputs.size(-1))
        targets = batch["tgt"][:, 1:].reshape(-1)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / max(len(dataloader), 1)
    if distributed:
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / dist.get_world_size()
    return avg_loss


def evaluate(model: Seq2Seq, dataloader: DataLoader, criterion, tokenizer, device) -> Dict:
    model.eval()
    total_loss = 0.0
    hyps = []
    refs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["src"], batch["src_lengths"], batch["tgt"], teacher_forcing_ratio=0.0)
            logits = outputs[:, 1:, :].reshape(-1, outputs.size(-1))
            targets = batch["tgt"][:, 1:].reshape(-1)
            loss = criterion(logits, targets)
            total_loss += loss.item()

            predictions = model.greedy_decode(batch["src"], batch["src_lengths"])
            for pred_ids, tgt_ids in zip(predictions, batch["tgt"]):
                hyp = tokenizer.decode_tgt(pred_ids.tolist())
                ref = tokenizer.decode_tgt(tgt_ids.tolist()[1:])  # skip BOS
                hyps.append(hyp)
                refs.append(ref)

    bleu = corpus_bleu(hyps, [refs]).score if hyps else 0.0
    return {"loss": total_loss / len(dataloader), "bleu": bleu}


def save_checkpoint(state: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser(description="Train RNN NMT model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--distributed", action="store_true", help="Force distributed mode (otherwise inferred from WORLD_SIZE)")
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
        world_size = 1

    tokenizer = load_tokenizer(config["data"]["tokenizer_file"])
    dataloaders = build_dataloaders(config, tokenizer, distributed, world_size, rank)

    model_cfg = Seq2SeqConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config["model"].get("embedding_dim", 256),
        hidden_dim=config["model"].get("hidden_dim", 512),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.3),
        pad_idx=tokenizer.pad_id,
        bos_idx=tokenizer.bos_id,
        eos_idx=tokenizer.eos_id,
        rnn_type=config["model"].get("architecture", "gru"),
        attention=config["model"].get("attention", "dot"),
        max_decode_len=config["model"].get("max_decode_len", 120),
    )
    model = Seq2Seq(model_cfg).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.tgt_pad_id)
    optimizer = optim.Adam(model.parameters(), lr=config["training"].get("learning_rate", 1e-3))

    teacher_forcing = config["training"].get("teacher_forcing_ratio", 1.0)
    tf_decay = config["training"].get("teacher_forcing_decay", 0.0)
    grad_clip = config["training"].get("max_grad_norm", 1.0)

    best_bleu = 0.0
    history = []
    for epoch in range(1, config["training"].get("num_epochs", 10) + 1):
        sampler = dataloaders.get("train_sampler")
        if distributed and sampler is not None:
            sampler.set_epoch(epoch)
        start = time.time()
        train_loss = train_epoch(model, dataloaders["train"], optimizer, criterion, device, teacher_forcing, grad_clip, distributed)
        eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        metrics = evaluate(eval_model, dataloaders["valid"], criterion, tokenizer, device) if rank == 0 else {"loss": 0.0, "bleu": 0.0}
        duration = time.time() - start
        if rank == 0:
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": metrics["loss"],
                "bleu": metrics["bleu"],
                "teacher_forcing": teacher_forcing,
                "time_sec": duration,
            })
            print(json.dumps(history[-1], ensure_ascii=False))

            if metrics["bleu"] > best_bleu:
                best_bleu = metrics["bleu"]
                save_path = Path(config["evaluation"].get("model_save_path", "experiments/logs/best_rnn.pt"))
                save_checkpoint({
                    "model_state": eval_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config,
                    "tokenizer": config["data"]["tokenizer_file"],
                    "epoch": epoch,
                    "bleu": best_bleu,
                }, save_path)

        teacher_forcing = max(0.0, teacher_forcing - tf_decay)

    if rank == 0:
        metrics_path = Path(config.get("experiments", {}).get("metrics_file", "experiments/results/rnn_metrics.csv"))
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        pd.DataFrame(history).to_csv(metrics_path, index=False)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()