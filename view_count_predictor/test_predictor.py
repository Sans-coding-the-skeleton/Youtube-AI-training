import torch
import pandas as pd
import numpy as np
from dataset_utils import YouTubeDataset, clean_text
from model import MultiModalViewCountModel

def main():
    # 1. Load Model and Metadata
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load("view_predictor.pth", map_location=device, weights_only=False)
    
    vocab = checkpoint['vocab']
    cat_to_idx = checkpoint['cat_to_idx']
    categories = checkpoint['categories']
    num_mean = checkpoint['num_mean']
    num_std = checkpoint['num_std']
    
    model = MultiModalViewCountModel(num_cats=len(categories), vocab_size=len(vocab)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 2. Load Dataset for samples
    full_dataset = YouTubeDataset("../dataset.csv", vocab=vocab)
    samples = full_dataset.df.sample(5)
    
    print(f"{'Title':<40} | {'Actual Views':<12} | {'Predicted Views':<12}")
    print("-" * 70)
    
    for _, row in samples.iterrows():
        title_str = str(row['title'])
        title_disp = (title_str[:37] + '...') if len(title_str) > 40 else title_str
        actual_views = int(row['view_count'])
        
        # Prepare input
        # num_cols = ['duration', 'month', 'day_of_week', 'title_len']
        upload_date = pd.to_datetime(row['upload_date'], format='%Y%m%d', errors='coerce')
        month = upload_date.month if pd.notnull(upload_date) else 0
        dow = upload_date.dayofweek if pd.notnull(upload_date) else 0
        title_len = len(clean_text(title_str).split())
        
        raw_num = np.array([row['duration'], month, dow, title_len], dtype=np.float32)
        norm_num = (raw_num - num_mean) / num_std
        
        num_f = torch.tensor([norm_num], dtype=torch.float32).to(device)
        cat_f = torch.tensor([cat_to_idx.get(str(row['categories']), 0)], dtype=torch.long).to(device)
        
        title_tokens = [vocab.get(w, 0) for w in clean_text(title_str).split()][:20]
        titles_p = torch.tensor([title_tokens], dtype=torch.long).to(device)
        
        with torch.no_grad():
            pred_log = model(num_f, cat_f, titles_p).item()
            pred_views = int(np.expm1(pred_log))
            
        print(f"{title_disp:<40} | {actual_views:<12} | {pred_views:<12}")

if __name__ == "__main__":
    main()
