"""
server.py
Flask backend for the Virality Predictor web app.
Serves the HTML frontend and exposes /api/predict endpoint.

Usage:
  python server.py
Then open http://localhost:5000
"""

import datetime
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
import yt_dlp
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from dataset import clean_text, VAL_TRANSFORMS
from model import ViralityNet

# ── Setup ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
APP_DIR    = BASE_DIR / "app"
MODEL_PATH = BASE_DIR / "virality_model.pth"

app = Flask(__name__, static_folder=str(APP_DIR), static_url_path="")

# Lazy-loaded model (loaded once on first request)
_model = None
_ckpt  = None


def get_model():
    global _model, _ckpt
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train.py first."
            )
        _ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        _model = ViralityNet(
            num_cats=len(_ckpt["cat_to_idx"]),
            vocab_size=len(_ckpt["vocab"]),
        )
        _model.load_state_dict(_ckpt["model_state_dict"])
        _model.eval()
        print("Model loaded.")
    return _model, _ckpt


# ── Log-score percentile (based on dataset distributions) ─────────────────────
# Approximate percentile breakpoints from training data (log1p view counts):
#  p10 ≈ 9.2, p25 ≈ 11.1, p50 ≈ 12.7, p75 ≈ 14.1, p90 ≈ 16.1
PERCENTILE_BREAKS = [0, 9.2, 11.1, 12.7, 14.1, 16.1, 99]
PERCENTILE_VALUES = [0,  10,   25,   50,   75,   90, 100]

def log_to_percentile(log_score: float) -> float:
    for i in range(len(PERCENTILE_BREAKS) - 1):
        lo, hi = PERCENTILE_BREAKS[i], PERCENTILE_BREAKS[i + 1]
        if lo <= log_score <= hi:
            p_lo, p_hi = PERCENTILE_VALUES[i], PERCENTILE_VALUES[i + 1]
            t = (log_score - lo) / (hi - lo + 1e-9)
            return round(p_lo + t * (p_hi - p_lo), 1)
    return 99.0 if log_score > PERCENTILE_BREAKS[-2] else 0.0


def format_views(n: float) -> str:
    if n >= 1e9:  return f"{n/1e9:.2f}B"
    if n >= 1e6:  return f"{n/1e6:.1f}M"
    if n >= 1e3:  return f"{n/1e3:.0f}K"
    return str(int(n))


def view_range(log_score: float):
    """Return a human-readable view range (±0.5 in log space)."""
    low  = max(0, np.expm1(log_score - 0.5))
    high = np.expm1(log_score + 0.5)
    return f"{format_views(low)} – {format_views(high)}"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(APP_DIR), "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided."}), 400

    # ── Fetch metadata via yt-dlp ──────────────────────────────────────────
    try:
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        return jsonify({"error": f"Could not fetch video info: {e}"}), 400

    title       = info.get("title", "Unknown")
    duration    = float(info.get("duration", 0) or 0)
    upload_date = info.get("upload_date", "20200101") or "20200101"
    categories  = info.get("categories") or ["Unknown"]
    thumb_url   = info.get("thumbnail", "")
    actual_views = info.get("view_count")
    channel     = info.get("uploader", "Unknown")
    video_id    = info.get("id", "")

    cat_str = categories[0] if categories else "Unknown"

    try:
        d = datetime.datetime.strptime(upload_date, "%Y%m%d")
        month, dow = float(d.month), float(d.weekday())
        upload_display = d.strftime("%B %d, %Y")
    except Exception:
        month, dow = 6.0, 0.0
        upload_display = upload_date

    # ── Build model inputs ────────────────────────────────────────────────────
    try:
        model, ckpt = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    vocab      = ckpt["vocab"]
    cat_to_idx = ckpt["cat_to_idx"]
    num_mean   = ckpt["num_mean"]
    num_std    = ckpt["num_std"]

    title_clean = clean_text(title)
    title_len   = float(len(title_clean.split()))

    tokens  = [vocab.get(w, 0) for w in title_clean.split()][:20]
    title_t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    raw   = np.array([duration, month, dow, title_len], dtype=np.float32)
    norm  = (raw - num_mean) / num_std
    num_t = torch.tensor(norm).unsqueeze(0)

    cat_idx = cat_to_idx.get(cat_str, 0)
    cat_t   = torch.tensor([cat_idx], dtype=torch.long)

    try:
        resp  = requests.get(thumb_url, timeout=12)
        img   = Image.open(BytesIO(resp.content)).convert("RGB")
        img_t = VAL_TRANSFORMS(img).unsqueeze(0)
    except Exception:
        img_t = torch.zeros(1, 3, 224, 224)

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        log_pred = model(img_t, title_t, num_t, cat_t).item()

    pred_views  = float(np.expm1(log_pred))
    percentile  = log_to_percentile(log_pred)

    result = {
        "title":         title,
        "channel":       channel,
        "category":      cat_str,
        "duration":      int(duration),
        "upload_date":   upload_display,
        "thumbnail_url": thumb_url,
        "video_id":      video_id,
        "log_score":     round(log_pred, 4),
        "predicted_views": pred_views,
        "predicted_views_fmt": format_views(pred_views),
        "predicted_range": view_range(log_pred),
        "percentile":    percentile,
    }

    if actual_views:
        log_actual  = float(np.log1p(actual_views))
        log_error   = abs(log_pred - log_actual)
        result["actual_views"]     = actual_views
        result["actual_views_fmt"] = format_views(actual_views)
        result["log_error"]        = round(log_error, 4)

    return jsonify(result)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  YouTube Virality Predictor")
    print("  http://localhost:5000")
    print("=" * 55)
    if not MODEL_PATH.exists():
        print(f"\n[WARNING] No model found at {MODEL_PATH}")
        print("  Run:  python train.py   before starting the server.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
