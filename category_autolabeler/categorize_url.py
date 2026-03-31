import torch
import yt_dlp
import sys
import os
import pandas as pd
import numpy as np
from nlp_utils import clean_text, tokenize
from model import CategoryClassifier

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
                'description': info.get('description', '')
            }
        except Exception as e:
            print(f"Error fetching metadata: {e}")
            return None

def main():
    # 1. Load Model and Metadata
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "category_model.pth"
    if not os.path.exists(model_path):
        model_path = os.path.join("category_autolabeler", "category_model.pth")
        
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"Model file '{model_path}' not found. Please train the model first.")
        return

    vocab = checkpoint['vocab']
    categories = checkpoint['categories']
    
    # Pre-calculated parameters from training (embed_dim=64, rnn_hidden=128 default)
    model = CategoryClassifier(len(vocab), len(categories), embed_dim=64).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Get URL
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter YouTube Video URL: ").strip()

    if not url:
        print("No URL provided.")
        return

    # 3. Fetch Metadata
    print(f"Fetching metadata for {url}...")
    meta = get_video_metadata(url)
    if not meta:
        return

    print(f"\nTitle: {meta['title']}")
    
    # 4. Predict
    text = clean_text(meta['title'] + ' ' + meta['description'])
    tokens = torch.tensor([tokenize(text, vocab)[:30]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(tokens)
        pred_idx = output.argmax(1).item()
        pred_cat = categories[pred_idx]

    print("-" * 30)
    print(f"Predicted Category: {pred_cat}")
    print("-" * 30)

if __name__ == "__main__":
    main()
