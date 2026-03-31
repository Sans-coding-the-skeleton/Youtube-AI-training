"""
download_thumbnails.py
Downloads thumbnails from the existing dataset.csv thumbnail URLs.
Saves each image as thumbnails/{video_id}.jpg
Run this ONCE before training.
"""

import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

CSV_PATH = Path(__file__).parent.parent / "dataset.csv"
THUMBNAILS_DIR = Path(__file__).parent / "thumbnails"
MAX_WORKERS = 16
TIMEOUT = 12


def download_one(row):
    video_id = str(row["id"])
    url = str(row["thumbnail"])
    out_path = THUMBNAILS_DIR / f"{video_id}.jpg"

    if out_path.exists():
        return video_id, True, "cached"

    def try_url(u):
        resp = requests.get(u, timeout=TIMEOUT)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(out_path, "JPEG", quality=90)

    try:
        try_url(url)
        return video_id, True, "ok"
    except Exception:
        # Fallback to standard JPG URL
        fallback = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        try:
            try_url(fallback)
            return video_id, True, "fallback"
        except Exception as e:
            # Try hqdefault as last resort
            hq = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
            try:
                try_url(hq)
                return video_id, True, "hq_fallback"
            except Exception as e2:
                return video_id, False, str(e2)


def main():
    THUMBNAILS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    rows = [row for _, row in df.iterrows()]
    total = len(rows)
    print(f"Downloading thumbnails for {total} videos (workers={MAX_WORKERS})...")

    success, fail, cached = 0, 0, 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_one, row): row["id"] for row in rows}
        for i, future in enumerate(as_completed(futures)):
            vid_id, ok, msg = future.result()
            if ok:
                if msg == "cached":
                    cached += 1
                else:
                    success += 1
            else:
                fail += 1

            if (i + 1) % 200 == 0 or (i + 1) == total:
                pct = (i + 1) / total * 100
                print(
                    f"  [{i+1}/{total}] {pct:.0f}% | "
                    f"New: {success}  Cached: {cached}  Failed: {fail}"
                )

    print(f"\nDone! {success} new, {cached} already existed, {fail} failed.")
    if fail > 0:
        print("Failed thumbnails will be replaced by a blank gray image during training.")


if __name__ == "__main__":
    main()
