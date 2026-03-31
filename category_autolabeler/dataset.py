import pandas as pd
import torch
from torch.utils.data import Dataset
from nlp_utils import clean_text, tokenize

class CategoryDataset(Dataset):
    def __init__(self, csv_path, vocab=None, cat_to_idx=None):
        self.df = pd.read_csv(csv_path)
        
        # 1. Clean and combine text
        self.df['full_text'] = (self.df['title'].fillna('') + ' ' + self.df['description'].fillna('')).apply(clean_text)
        
        # 2. Handle labels
        if cat_to_idx is None:
            self.categories = sorted(self.df['categories'].fillna('Unknown').unique())
            self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        else:
            self.cat_to_idx = cat_to_idx
            self.categories = list(cat_to_idx.keys())
            
        self.df['label'] = self.df['categories'].fillna('Unknown').map(self.cat_to_idx)
        
        # 3. Tokenize and Truncate
        self.vocab = vocab
        self.tokenized_texts = [tokenize(t, vocab)[:30] for t in self.df['full_text']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.tokenized_texts[idx], dtype=torch.long)
        label = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.long)
        return tokens, label

from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        text_list.append(_text)
    
    label_list = torch.stack(label_list)
    # Pad sequences to the same length in the batch
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    
    return text_list, None, label_list
