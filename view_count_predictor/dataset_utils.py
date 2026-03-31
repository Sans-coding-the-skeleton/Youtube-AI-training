import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_vocab(texts, max_tokens=5000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {word: i + 1 for i, (word, _) in enumerate(counter.most_common(max_tokens))}
    vocab['<UNK>'] = 0
    return vocab

class YouTubeDataset(Dataset):
    def __init__(self, csv_path, vocab=None):
        self.df = pd.read_csv(csv_path)
        
        # 1. Text Preprocessing
        self.df['title_clean'] = self.df['title'].apply(clean_text)
        if vocab is None:
            self.vocab = build_vocab(self.df['title_clean'])
        else:
            self.vocab = vocab
            
        # 2. Date Features
        # upload_date is usually YYYYMMDD in yt-dlp metadata
        self.df['upload_date'] = pd.to_datetime(self.df['upload_date'], format='%Y%m%d', errors='coerce')
        self.df['month'] = self.df['upload_date'].dt.month.fillna(0)
        self.df['day_of_week'] = self.df['upload_date'].dt.dayofweek.fillna(0)
        
        # 3. Numerical Features
        self.df['duration'] = self.df['duration'].fillna(0)
        self.df['title_len'] = self.df['title_clean'].apply(lambda x: len(x.split()))
        
        # 4. Target: log scale view_count
        self.df['target'] = np.log1p(self.df['view_count'].fillna(0))
        
        # 5. Categorical: categories
        self.categories = sorted(self.df['categories'].fillna('None').unique())
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        
        # Normalize Numerical
        num_cols = ['duration', 'month', 'day_of_week', 'title_len']
        self.features_num = self.df[num_cols].values.astype(np.float32)
        self.num_mean = self.features_num.mean(axis=0)
        self.num_std = self.features_num.std(axis=0) + 1e-6
        self.features_num = (self.features_num - self.num_mean) / self.num_std
        
        self.targets = self.df['target'].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cat_str = str(self.df.iloc[idx]['categories'])
        cat_idx = self.cat_to_idx.get(cat_str, 0)
        
        title_tokens = [self.vocab.get(w, 0) for w in self.df.iloc[idx]['title_clean'].split()][:20]
        
        return {
            'num_features': torch.tensor(self.features_num[idx]),
            'cat_feature': torch.tensor(cat_idx, dtype=torch.long),
            'title_tokens': torch.tensor(title_tokens, dtype=torch.long),
            'target': torch.tensor(self.targets[idx])
        }

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    num_f = torch.stack([item['num_features'] for item in batch])
    cat_f = torch.stack([item['cat_feature'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    
    titles = [item['title_tokens'] for item in batch]
    titles_padded = pad_sequence(titles, batch_first=True, padding_value=0)
    
    return {
        'num_features': num_f,
        'cat_feature': cat_f,
        'title_tokens': titles_padded,
        'target': targets
    }
