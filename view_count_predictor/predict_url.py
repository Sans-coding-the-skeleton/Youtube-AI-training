import torch
import yt_dlp
import sys
import os
import pandas as pd
import numpy as np
from dataset_utils import clean_text
from model import MultiModalViewCountModel

def get_video_metadata(url):
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', ''),
                'duration': info.get('duration', 0),
                'upload_date': info.get('upload_date', ''),
                'categories': info.get('categories', ['None'])[0]
            }
        except Exception as e:
            print(f"Error fetching metadata: {e}")
            return None

def main():
    # 1. Load Model and Metadata
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "view_predictor.pth"
    if not os.path.exists(model_path):
        model_path = os.path.join("view_count_predictor", "view_predictor.pth")
        
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"Model file '{model_path}' not found. Please train the model first.")
        return

    vocab = checkpoint['vocab']
    cat_to_idx = checkpoint['cat_to_idx']
    categories = checkpoint['categories']
    num_mean = checkpoint['num_mean']
    num_std = checkpoint['num_std']
    
    model = MultiModalViewCountModel(num_cats=len(categories), vocab_size=len(vocab)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Get URL
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter YouTube Video URL: ").strip()

    if not url:
        return

    # 3. Fetch Metadata
    print(f"Fetching metadata for {url}...")
    meta = get_video_metadata(url)
    if not meta:
        return

    print(f"\nTitle: {meta['title']}")
    
    # 4. Prepare Features
    upload_date = pd.to_datetime(meta['upload_date'], format='%Y%m%d', errors='coerce')
    month = upload_date.month if pd.notnull(upload_date) else 0
    dow = upload_date.dayofweek if pd.notnull(upload_date) else 0
    title_len = len(clean_text(meta['title']).split())
    
    raw_num = np.array([meta['duration'], month, dow, title_len], dtype=np.float32)
    norm_num = (raw_num - num_mean) / num_std
    
    num_f = torch.tensor([norm_num], dtype=torch.float32).to(device)
    cat_f = torch.tensor([cat_to_idx.get(str(meta['categories']), 0)], dtype=torch.long).to(device)
    
    title_tokens = [vocab.get(w, 0) for w in clean_text(meta['title']).split()][:20]
    titles_p = torch.tensor([title_tokens], dtype=torch.long).to(device)
    
    # 5. Predict
    with torch.no_grad():
        pred_log = model(num_f, cat_f, titles_p).item()
        pred_views = int(np.expm1(pred_log))

    print("-" * 30)
    print(f"Predicted Views: {pred_views:,}")
    print("-" * 30)

if __name__ == "__main__":
    main()
