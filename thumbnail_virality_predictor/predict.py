"""
predict.py
CLI tool to predict virality from a YouTube URL.
Usage: python predict.py <youtube_url>
"""

import sys
import datetime
import numpy as np
import requests
import torch
import yt_dlp
from io import BytesIO
from pathlib import Path
from PIL import Image

from dataset import clean_text, VAL_TRANSFORMS
from model import ViralityNet

MODEL_PATH = Path(__file__).parent / "virality_model.pth"


def load_model(model_path=MODEL_PATH):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ViralityNet(
        num_cats=len(ckpt["cat_to_idx"]),
        vocab_size=len(ckpt["vocab"]),
        num_channels=len(ckpt["channel_to_idx"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def fetch_info(url: str) -> dict:
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)


def build_tensors(info: dict, ckpt: dict):
    vocab      = ckpt["vocab"]
    cat_to_idx = ckpt["cat_to_idx"]
    chan_to_idx = ckpt["channel_to_idx"]
    num_mean   = ckpt["num_mean"]
    num_std    = ckpt["num_std"]

    title       = info.get("title", "")
    duration    = float(info.get("duration", 0) or 0)
    upload_date = info.get("upload_date", "20200101") or "20200101"
    categories  = info.get("categories", ["Unknown"]) or ["Unknown"]
    thumb_url   = info.get("thumbnail", "")
    channel     = info.get("uploader", "Unknown")
    sub_count   = float(info.get("channel_follower_count", 0) or 0)

    try:
        d   = datetime.datetime.strptime(upload_date, "%Y%m%d")
        month, dow = float(d.month), float(d.weekday())
    except Exception:
        month, dow = 6.0, 0.0

    title_clean = clean_text(title)
    title_len   = float(len(title_clean.split()))
    cat_str     = categories[0] if categories else "Unknown"

    # Title tokens
    tokens = [vocab.get(w, 0) for w in title_clean.split()][:20]
    title_t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    # Numerical (normalized)
    raw = np.array([duration, month, dow, title_len, sub_count], dtype=np.float32)
    norm = (raw - num_mean) / num_std
    num_t = torch.tensor(norm).unsqueeze(0)

    # Category
    cat_idx = cat_to_idx.get(cat_str, 0)
    cat_t = torch.tensor([cat_idx], dtype=torch.long)

    # Channel
    chan_idx = chan_to_idx.get(channel, 0)
    chan_t = torch.tensor([chan_idx], dtype=torch.long)

    # Image
    try:
        resp  = requests.get(thumb_url, timeout=12)
        img   = Image.open(BytesIO(resp.content)).convert("RGB")
        img_t = VAL_TRANSFORMS(img).unsqueeze(0)
    except Exception:
        img_t = torch.zeros(1, 3, 224, 224)

    return img_t, title_t, num_t, cat_t, chan_t, cat_str, title


def format_views(n):
    if n >= 1e9:  return f"{n/1e9:.2f}B"
    if n >= 1e6:  return f"{n/1e6:.2f}M"
    if n >= 1e3:  return f"{n/1e3:.0f}K"
    return str(int(n))


def predict(url: str):
    if not MODEL_PATH.exists():
        print(f"[ERROR] No trained model found at {MODEL_PATH}")
        print("Run train.py first.")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}...")
    model, ckpt = load_model()

    print(f"Fetching video info for: {url}")
    info = fetch_info(url)

    img_t, title_t, num_t, cat_t, chan_t, cat_str, title = build_tensors(info, ckpt)

    with torch.no_grad():
        log_pred = model(img_t, title_t, num_t, cat_t, chan_t).item()

    pred_views   = np.expm1(log_pred)
    actual_views = info.get("view_count")

    print("\n" + "=" * 50)
    print(f"  Title     : {title}")
    print(f"  Category  : {cat_str}")
    print(f"  Duration  : {info.get('duration', '?')}s")
    print(f"  Log score : {log_pred:.4f}")
    print(f"  Predicted : ~{format_views(pred_views)} views")
    if actual_views:
        log_actual = np.log1p(actual_views)
        error = abs(log_pred - log_actual)
        print(f"  Actual    : {format_views(actual_views)} views  (log error: {error:.3f})")
    print("=" * 50)

    return {
        "title": title,
        "category": cat_str,
        "predicted_views": pred_views,
        "log_score": log_pred,
        "actual_views": actual_views,
        "thumbnail_url": info.get("thumbnail", ""),
    }


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else input("YouTube URL: ").strip()
    predict(url)
