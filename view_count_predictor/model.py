import torch
import torch.nn as nn

class MultiModalViewCountModel(nn.Module):
    def __init__(self, num_cats, vocab_size, cat_emb_dim=16, text_emb_dim=64, rnn_hidden=64, meta_dim=4, hidden_dim=128):
        super(MultiModalViewCountModel, self).__init__()
        
        # 1. Categorical Embedding
        self.cat_embedding = nn.Embedding(num_cats, cat_emb_dim)
        
        # 2. Text Encoder (GRU)
        self.text_embedding = nn.Embedding(vocab_size, text_emb_dim, padding_idx=0)
        self.rnn = nn.GRU(text_emb_dim, rnn_hidden, batch_first=True, bidirectional=False)
        
        # 3. Metadata Sub-network
        # meta_dim: duration, month, day_of_week, title_len
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_dim + cat_emb_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 4. Fusion Layer
        # Concatenate: rnn_hidden + 32
        self.fusion_fc = nn.Sequential(
            nn.Linear(rnn_hidden + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Predicting log(view_count + 1)
        )
        
    def forward(self, num_features, cat_feature, title_tokens):
        # Text encoding
        text_emb = self.text_embedding(title_tokens)
        _, h_n = self.rnn(text_emb)
        # h_n shape: [1, batch, rnn_hidden]
        text_features = h_n.squeeze(0)
        
        # Metadata encoding
        cat_emb = self.cat_embedding(cat_feature)
        meta_input = torch.cat([num_features, cat_emb], dim=1)
        meta_features = self.meta_fc(meta_input)
        
        # Fusion
        combined = torch.cat([text_features, meta_features], dim=1)
        x = self.fusion_fc(combined)
        return x.squeeze()
