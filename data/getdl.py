import os
import torch
import gdown
import pickle
import pandas as pd

from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from utils.preprocessing import preprocess_text
from utils.glove_embeddings import load_glove_embeddings

def build_vocab(texts):
    all_tokens = [token for text in texts for token in text]
    unique_tokens = list(set(all_tokens))
    vocab = {word: idx + 2 for idx, word in enumerate(unique_tokens)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def save_vocab(vocab, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)
        
def load_vocab_from_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
class TEXTDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        indices = [self.vocab.get(t, 1) for t in tokens]
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        label = self.labels[idx]
        return torch.tensor(indices), torch.tensor(label)

def split_and_create_loaders(texts, labels, vocab, glove_path, batch_size, embedding_dim, max_len):
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42)

    train_dataset = TEXTDataset(train_texts, train_labels, vocab, max_len)
    test_dataset = TEXTDataset(test_texts, test_labels, vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    embedding_weights = load_glove_embeddings(glove_path, embedding_dim, vocab)

    return train_loader, test_loader, vocab, embedding_weights

def prepare_imdb_dataloaders(csv_path, glove_path, batch_size=64, embedding_dim=100, max_len=100):
    df = pd.read_csv(csv_path)
    raw_texts = df['review'].tolist()
    label_map = {'positive': 1, 'negative': 0}
    labels = [label_map[l] for l in df['sentiment'].tolist()]

    print('Preprocessing IMDB texts...')
    texts_clean = [preprocess_text(t, use_spell=False) for t in tqdm(raw_texts)]
    vocab = build_vocab(texts_clean)
    save_vocab(vocab, './models/checkpoints/sentiment_analysis_vocab.pkl')
    
    return split_and_create_loaders(texts_clean, labels, vocab, glove_path, batch_size, embedding_dim, max_len)

def prepare_sms_dataloaders(csv_path, glove_path, batch_size=64, embedding_dim=100, max_len=100):
    df = pd.read_csv(csv_path, encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    label_map = {'ham': 0, 'spam': 1}
    labels = [label_map[l] for l in df['label'].tolist()]
    raw_texts = df['text'].tolist()

    print('Preprocessing SMS texts...')
    texts_clean = [preprocess_text(t, use_spell=False) for t in tqdm(raw_texts)]
    vocab = build_vocab(texts_clean)
    save_vocab(vocab, './models/checkpoints/spam_detection_vocab.pkl')
    
    return split_and_create_loaders(texts_clean, labels, vocab, glove_path, batch_size, embedding_dim, max_len)

def prepare_newsgroups_dataloaders(glove_path, batch_size=64, embedding_dim=100, max_len=200):
    print("Loading 20 Newsgroups data...")
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    texts = data.data
    labels = data.target
    num_classes = len(data.target_names)

    print("Preprocessing Newsgroups texts...")
    texts_clean = [preprocess_text(t, use_spell=False) for t in tqdm(texts)]
    vocab = build_vocab(texts_clean)
    save_vocab(vocab, './models/checkpoints/topic_classification_vocab.pkl')
    
    train_loader, test_loader, vocab, embedding_weights = split_and_create_loaders(
        texts_clean, labels, vocab, glove_path, batch_size, embedding_dim, max_len)

    return train_loader, test_loader, vocab, embedding_weights, num_classes

def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {output_path}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{output_path} already exists.")

def download_all_models():
    files = {
        "models/checkpoints/SMS_LSTM.pt": "1TcSty71gvSfAxVAmxjCZp5UFI40KQL9m",
        "models/checkpoints/SMS_LSTMAttn.pt": "1UJAX-L1aFso9M1oepwnhZ_g6Zxio4OGG",
        "models/checkpoints/IMDB_Logistic_Regression.pkl": "1K4AhgG1-JaQrue6CS4c7MBrUQfCnnGTd",
        "models/checkpoints/SMS_Spam_Naive_Bayes.pkl": "13UNgKs_mnIUFEQRxGXwtM5NqZVFm2NBd",
        "models/checkpoints/20_Newsgroups_Naive_Bayes.pkl": "1wHMJSxLo4g9z8HJn8IeWlOctROK96LRI",
        "models/checkpoints/spam_detection_vocab.pkl": "17ikZ7T0bmGgJpCGUzMqmIbrEn807Splf"
    }

    for path, file_id in files.items():
        download_model(file_id, path)
        

if __name__ == "__main__":
    download_all_models()