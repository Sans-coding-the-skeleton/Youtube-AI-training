import torch
from nlp_utils import clean_text, tokenize
from model import CategoryClassifier

def main():
    # 1. Load Model and Metadata
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load("category_model.pth", map_location=device, weights_only=False)
    vocab = checkpoint['vocab']
    cat_to_idx = checkpoint['cat_to_idx']
    categories = checkpoint['categories']
    
    # Initialize model with the same parameters as used in training (embed_dim=64, rnn_hidden=128 default)
    model = CategoryClassifier(len(vocab), len(categories), embed_dim=64).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 2. Test Cases
    test_texts = [
        "How to build a gaming PC 2024 tutorial",
        "Funny cat videos compilation 2024",
        "Advanced Python programming for data science",
        "Exploring abandoned places in the desert",
        "Tesla Model 3 review and test drive",
        "Minecraft survival series episode 1"
    ]
    
    print(f"{'Text Snippet':<45} | {'Predicted Category':<20}")
    print("-" * 75)
    
    for text in test_texts:
        cleaned = clean_text(text)
        tokens = torch.tensor([tokenize(cleaned, vocab)[:30]], dtype=torch.long).to(device)
        
        with torch.no_grad():
            output = model(tokens)
            pred_idx = output.argmax(1).item()
            pred_cat = categories[pred_idx]
            
        snippet = (text[:42] + '...') if len(text) > 45 else text
        print(f"{snippet:<45} | {pred_cat:<20}")

if __name__ == "__main__":
    main()
