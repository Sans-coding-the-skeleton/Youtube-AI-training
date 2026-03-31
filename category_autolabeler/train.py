import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
from nlp_utils import clean_text, build_vocab
from dataset import CategoryDataset, collate_batch
from model import CategoryClassifier
import os

# Hyperparameters
CSV_PATH = "../dataset.csv"
EMBED_DIM = 64
RNN_HIDDEN = 64
BATCH_SIZE = 128
LEARNING_RATE = 0.002
EPOCHS = 15

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_acc, total_count = 0, 0
    for text, _, label in loader:
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
    return total_acc / total_count

def evaluate(model, loader):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for text, _, label in loader:
            text, label = text.to(device), label.to(device)
            predicted_label = model(text)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

def main():
    # 1. Prepare VOCAB and Categories
    print("Preparing Vocabulary...")
    df = pd.read_csv(CSV_PATH)
    texts = (df['title'].fillna('') + ' ' + df['description'].fillna('')).apply(clean_text)
    vocab = build_vocab(texts)
    
    # 2. Load Dataset
    dataset = CategoryDataset(CSV_PATH, vocab=vocab)
    num_classes = len(dataset.categories)
    print(f"Dataset loaded. Classes: {num_classes}, Vocab Size: {len(vocab)}")
    
    # 3. Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    
    # 4. Model
    model = CategoryClassifier(len(vocab), num_classes, EMBED_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        acc_train = train_epoch(model, train_loader, optimizer, criterion)
        acc_val = evaluate(model, test_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Train Acc: {acc_train:.4f} | Val Acc: {acc_val:.4f}")
            
    print("Training finished.")
    
    # 6. Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'cat_to_idx': dataset.cat_to_idx,
        'categories': dataset.categories
    }, "category_model.pth")
    print("Model saved to category_model.pth")

if __name__ == "__main__":
    main()
