import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset_utils import YouTubeDataset, collate_fn
from model import MultiModalViewCountModel
import matplotlib.pyplot as plt
import os

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.002
EPOCHS = 15
CSV_PATH = "../dataset.csv"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    # 1. Load Dataset
    print("Loading dataset...")
    full_dataset = YouTubeDataset(CSV_PATH)
    vocab_size = len(full_dataset.vocab)
    num_cats = len(full_dataset.categories)
    
    # 2. Split into Train/Test
    test_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 3. Initialize Model, Loss, Optimizer
    model = MultiModalViewCountModel(num_cats=num_cats, vocab_size=vocab_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    test_losses = []
    
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            num_f = batch['num_features'].to(device)
            cat_f = batch['cat_feature'].to(device)
            titles = batch['title_tokens'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(num_f, cat_f, titles)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                num_f = batch['num_features'].to(device)
                cat_f = batch['cat_feature'].to(device)
                titles = batch['title_tokens'].to(device)
                target = batch['target'].to(device)
                outputs = model(num_f, cat_f, titles)
                loss = criterion(outputs, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        test_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # 4. Evaluation
    print("Training finished.")
    
    # Save the model and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': full_dataset.vocab,
        'cat_to_idx': full_dataset.cat_to_idx,
        'categories': full_dataset.categories,
        'num_mean': full_dataset.num_mean,
        'num_std': full_dataset.num_std
    }, "view_predictor.pth")
    print("Model saved to view_predictor.pth")

if __name__ == "__main__":
    main()
