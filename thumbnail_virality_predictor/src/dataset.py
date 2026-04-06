"""
dataset.py
ViralityDataset — loads thumbnails + title + metadata for each YouTube video.
Inputs to model:
  - image:        thumbnail (224x224, ImageNet normalized)
  - title_tokens: tokenized & truncated title (max 20 words)
  - num_features: [duration, month, day_of_week, title_len] (z-score normalized)
  - cat_feature:  category index
Target: log1p(view_count)
"""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

THUMBNAILS_DIR = Path(__file__).parent / "thumbnails"
MAX_TITLE_LEN = 20
MAX_VOCAB = 5000

# ── Transforms ──────────────────────────────────────────────────────────────

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Blank gray placeholder for missing thumbnails
_BLANK_PIL = Image.new("RGB", (224, 224), (128, 128, 128))
BLANK_TENSOR = VAL_TRANSFORMS(_BLANK_PIL)


# ── Text utilities ───────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocab(texts, max_tokens=MAX_VOCAB):
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    vocab = {word: i + 1 for i, (word, _) in enumerate(counter.most_common(max_tokens))}
    vocab["<UNK>"] = 0
    return vocab


# ── Dataset ──────────────────────────────────────────────────────────────────

class ViralityDataset(Dataset):
    def __init__(
        self,
        csv_path,
        vocab=None,
        cat_to_idx=None,
        channel_to_idx=None,
        is_train=True,
        num_mean=None,
        num_std=None,
        indices=None,
    ):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["view_count"]).reset_index(drop=True)

        # Optionally work on a subset (for train/val split)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        self.is_train = is_train
        self.transform = TRAIN_TRANSFORMS if is_train else VAL_TRANSFORMS

        # ── Text ────────────────────────────────────────────────────────────
        df["title_clean"] = df["title"].apply(clean_text)
        if vocab is None:
            self.vocab = build_vocab(df["title_clean"])
        else:
            self.vocab = vocab

        # ── Date features ────────────────────────────────────────────────────
        df["upload_date"] = pd.to_datetime(df["upload_date"], format="%Y%m%d", errors="coerce")
        df["month"] = df["upload_date"].dt.month.fillna(6).astype(float)
        df["day_of_week"] = df["upload_date"].dt.dayofweek.fillna(0).astype(float)

        # ── Numerical features ───────────────────────────────────────────────
        df["duration"] = df["duration"].fillna(0).astype(float)
        df["title_len"] = df["title_clean"].apply(lambda x: len(x.split())).astype(float)
        df["channel_follower_count"] = df["channel_follower_count"].fillna(0).astype(float)

        num_cols = ["duration", "month", "day_of_week", "title_len", "channel_follower_count"]
        feats = df[num_cols].values.astype(np.float32)
        if num_mean is None:
            self.num_mean = feats.mean(axis=0)
            self.num_std = feats.std(axis=0) + 1e-6
        else:
            self.num_mean = num_mean
            self.num_std = num_std
        self.features_num = (feats - self.num_mean) / self.num_std

        # ── Categories ───────────────────────────────────────────────────────
        if cat_to_idx is None:
            self.categories = sorted(df["categories"].fillna("Unknown").unique())
            self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        else:
            self.cat_to_idx = cat_to_idx
            self.categories = list(cat_to_idx.keys())

        # ── Channels ─────────────────────────────────────────────────────────
        if channel_to_idx is None:
            self.channels = sorted(df["channel"].fillna("Unknown").unique())
            self.channel_to_idx = {ch: i for i, ch in enumerate(self.channels)}
        else:
            self.channel_to_idx = channel_to_idx
            self.channels = list(channel_to_idx.keys())

        # ── Target ───────────────────────────────────────────────────────────
        self.targets = np.log1p(df["view_count"].values).astype(np.float32)

        self.df = df  # keep for __getitem__ access

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        img_path = THUMBNAILS_DIR / f"{row['id']}.jpg"
        if img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img)
            except Exception:
                img_tensor = BLANK_TENSOR
        else:
            img_tensor = BLANK_TENSOR

        # Title tokens (truncated)
        tokens = [self.vocab.get(w, 0) for w in row["title_clean"].split()][:MAX_TITLE_LEN]

        # Category
        cat_str = str(row["categories"])
        cat_idx = self.cat_to_idx.get(cat_str, 0)

        # Channel
        chan_str = str(row["channel"])
        chan_idx = self.channel_to_idx.get(chan_str, 0)

        return {
            "image": img_tensor,
            "title_tokens": torch.tensor(tokens, dtype=torch.long),
            "num_features": torch.tensor(self.features_num[idx]),
            "cat_feature": torch.tensor(cat_idx, dtype=torch.long),
            "channel_feature": torch.tensor(chan_idx, dtype=torch.long),
            "target": torch.tensor(self.targets[idx]),
        }


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "title_tokens": pad_sequence(
            [b["title_tokens"] for b in batch], batch_first=True, padding_value=0
        ),
        "num_features": torch.stack([b["num_features"] for b in batch]),
        "cat_feature": torch.stack([b["cat_feature"] for b in batch]),
        "channel_feature": torch.stack([b["channel_feature"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
    }
