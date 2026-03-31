import torch
import torch.nn as nn

class CategoryClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=128, rnn_hidden=128, dropout=0.3):
        super(CategoryClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Bi-directional GRU
        self.rnn = nn.GRU(embed_dim, rnn_hidden, batch_first=True, bidirectional=True)
        
        # Regularization and Output
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(rnn_hidden * 2) # *2 for bidirectional
        
        self.fc1 = nn.Linear(rnn_hidden * 2, rnn_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(rnn_hidden, num_classes)
        
    def forward(self, text, offsets=None):
        # text: [batch_size, seq_len] or packed tokens
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        
        # RNN layer
        rnn_out, _ = self.rnn(embedded)
        
        # Global max pooling across time dimension
        # rnn_out: [batch_size, seq_len, hidden*2]
        x = torch.max(rnn_out, dim=1)[0]
        
        x = self.batch_norm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
