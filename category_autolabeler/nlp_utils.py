import re
from collections import Counter

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove everything except alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_vocab(texts, max_tokens=10000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    
    # Most common tokens
    vocab = {word: i + 1 for i, (word, _) in enumerate(counter.most_common(max_tokens))}
    vocab['<UNK>'] = 0 # Unknown token
    return vocab

def tokenize(text, vocab):
    tokens = [vocab.get(word, 0) for word in text.split()]
    return tokens
