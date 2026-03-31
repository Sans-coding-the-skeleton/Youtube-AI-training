"""
train.py
Trains ViralityNet on the YouTube dataset (thumbnail + title + metadata).
Saves the best checkpoint to virality_model.pth.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import ViralityDataset, collate_fn
from model import ViralityNet

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH     = "../dataset.csv"
MODEL_OUT    = "virality_model.pth"
PLOT_OUT     = "training_plot.png"
BATCH_SIZE   = 64
EPOCHS       = 25
LR           = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.2
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ──────────────────────────────────────────────────────────────────

def format_views(n):
    """Convert a number to a human-readable string."""
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(int(n))


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    preds, targets = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            img      = batch["image"].to(DEVICE)
            tokens   = batch["title_tokens"].to(DEVICE)
            num_f    = batch["num_features"].to(DEVICE)
            cat_f    = batch["cat_feature"].to(DEVICE)
            chan_f   = batch["channel_feature"].to(DEVICE)
            target   = batch["target"].to(DEVICE)

            out  = model(img, tokens, num_f, cat_f, chan_f)
            loss = criterion(out, target)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            preds.extend(out.detach().cpu().numpy())
            targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(loader)
    mae = float(np.mean(np.abs(np.array(preds) - np.array(targets))))
    return avg_loss, mae, np.array(preds), np.array(targets)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Split indices (before loading images) ────────────────────────────────
    import pandas as pd
    df_full = pd.read_csv(CSV_PATH).dropna(subset=["view_count"])
    n = len(df_full)
    idx = np.random.permutation(n)
    val_n = int(VAL_SPLIT * n)
    val_idx   = idx[:val_n].tolist()
    train_idx = idx[val_n:].tolist()
    print(f"Dataset: {n} records → Train: {len(train_idx)}, Val: {len(val_idx)}")

    # ── Build vocab/stats on full dataset ────────────────────────────────────
    print("Building vocabulary and stats...")
    ref_ds = ViralityDataset(CSV_PATH, is_train=False)
    vocab      = ref_ds.vocab
    cat_to_idx = ref_ds.cat_to_idx
    categories = ref_ds.categories
    channel_to_idx = ref_ds.channel_to_idx
    channels   = ref_ds.channels
    num_mean   = ref_ds.num_mean
    num_std    = ref_ds.num_std

    # ── Create train / val subsets with correct transforms ───────────────────
    train_ds = ViralityDataset(
        CSV_PATH, vocab=vocab, cat_to_idx=cat_to_idx, channel_to_idx=channel_to_idx,
        is_train=True,  num_mean=num_mean, num_std=num_std, indices=train_idx,
    )
    val_ds = ViralityDataset(
        CSV_PATH, vocab=vocab, cat_to_idx=cat_to_idx, channel_to_idx=channel_to_idx,
        is_train=False, num_mean=num_mean, num_std=num_std, indices=val_idx,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=(DEVICE.type == "cuda"),
    )

    # ── Model, optimizer, scheduler ──────────────────────────────────────────
    model = ViralityNet(
        num_cats=len(cat_to_idx),
        vocab_size=len(vocab),
        num_channels=len(channel_to_idx),
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Parameters — trainable: {trainable:,}  frozen: {frozen:,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    print(f"\nStarting training for {EPOCHS} epochs...\n")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_mae, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_mae, val_preds, val_targets = run_epoch(model, val_loader, criterion)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Example prediction to show in human-readable form
        sample_pred   = format_views(np.expm1(val_preds[0]))
        sample_actual = format_views(np.expm1(val_targets[0]))

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"Train: {train_loss:.4f}  Val: {val_loss:.4f} | "
            f"MAE(log): {val_mae:.3f} | "
            f"Example: pred={sample_pred}  actual={sample_actual}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "cat_to_idx": cat_to_idx,
                    "categories": categories,
                    "channel_to_idx": channel_to_idx,
                    "channels": channels,
                    "num_mean": num_mean,
                    "num_std": num_std,
                },
                MODEL_OUT,
            )
            print(f"           ✓ New best saved (val={best_val_loss:.4f})")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train MSE")
    plt.plot(range(1, EPOCHS + 1), val_losses,   label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log-view-count space)")
    plt.title("ViralityNet Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=150)
    print(f"Saved loss plot to {PLOT_OUT}")


if __name__ == "__main__":
    main()
