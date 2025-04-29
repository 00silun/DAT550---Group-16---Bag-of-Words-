from gensim.models import FastText, KeyedVectors
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import re
from nltk.corpus import stopwords
from ffnn import FFNN
from evaluation import evaluate_model
from trainModel import train_model
import nltk
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from cleanText import clean_text
import os
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess
from pooling import Pooling
from gensim.models.fasttext import load_facebook_vectors
#nltk.download('stopwords') # Uncomment this line if stopwords are not downloaded
from embedding_loader import EmbeddingLoader
from gensim.models import FastText
from generate_custom_embedding import generateCustomEmbeddings
from build_embedding_layer import buildEmbeddingLayer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"    # Set to the GPU you want to use

stop_words = set(stopwords.words('english'))

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df['abstract'] = df['abstract'].astype(str).apply(clean_text)
    return df['abstract'].tolist(), df['label'].tolist()

# Vectorize and dataset creation
def vectorize_text(text, embeddings, dim, pooling_fn, **kwargs):
    words = text.split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    return pooling_fn(vectors, dim, **kwargs) if kwargs else pooling_fn(vectors, dim)

def create_dataset(texts, labels, embeddings, dim, pooling_fn, **kwargs):
    X = np.array([vectorize_text(t, embeddings, dim, pooling_fn, **kwargs) for t in texts])
    y = np.array(labels)
    return X, y

class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=100):
        self.sequences = []
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

        for text in texts:
            indices = [word2idx.get(word, word2idx['<UNK>']) for word in text.split()[:max_len]]
            # Pad
            while len(indices) < max_len:
                indices.append(word2idx['<PAD>'])
            self.sequences.append(indices)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# Main setup
if __name__ == "__main__":
    EMBEDDING_DIM = 300
    DATASET_PATH = "data/arxiv100.csv"
    EMBEDDING_TYPE = "fasttext"  # Options: fasttext, word2vec, glove, custom_word2vec, custom_fasttext
    FINE_TUNE = True  # Set to False if you want to freeze embeddings
    num_epochs = 20

    embedding_paths = {
        "fasttext": "embeddings/crawl-300d-2M-subword.bin",
        "word2vec": "embeddings/GoogleNews-vectors-negative300.bin.gz",
        "glove": "embeddings/glove.6B.300d.txt",
        "custom_word2vec": "embeddings/custom_word2vec.vec",
        "custom_fasttext": "embeddings/custom_fasttext.bin"
    }

    pooling_methods = {
        #"mean": Pooling.mean,
        "max": Pooling.max,
        #"sum": Pooling.sum,
        #"mean_max": Pooling.mean_max,
    }

    # Load dataset
    abstracts, labels = load_dataset(DATASET_PATH)
    label_to_idx = {label: i for i, label in enumerate(sorted(set(labels)))}
    y = [label_to_idx[label] for label in labels]

    # Load embeddings
    if EMBEDDING_TYPE == "fasttext":
        embeddings = EmbeddingLoader.load_fasttext(embedding_paths[EMBEDDING_TYPE])
    elif EMBEDDING_TYPE == "word2vec":
        embeddings = EmbeddingLoader.load_word2vec(embedding_paths[EMBEDDING_TYPE], binary=True)
    elif EMBEDDING_TYPE == "glove":
        embeddings = EmbeddingLoader.load_glove(embedding_paths[EMBEDDING_TYPE])
    elif EMBEDDING_TYPE == "custom_fasttext":
        embeddings = EmbeddingLoader.load_custom_fasttext(embedding_paths[EMBEDDING_TYPE])
    else:
        raise ValueError("Invalid EMBEDDING_TYPE selected.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if FINE_TUNE:
        print("Fine-tuning embeddings during training.")
        embedding_layer, word2idx = buildEmbeddingLayer(
            embeddings_dict=embeddings,
            embedding_dim=EMBEDDING_DIM,
            freeze=False
        )

        X_train, X_dev, y_train, y_dev = train_test_split(abstracts, y, test_size=0.2, stratify=y, random_state=42)
        train_dataset = TextDataset(X_train, y_train, word2idx)
        val_dataset = TextDataset(X_dev, y_dev, word2idx)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)

        model = FFNN(
            embedding_layer=embedding_layer,
            hidden_dims=[256, 128, 64],
            output_dim=len(label_to_idx)
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, device=device, name=f"{EMBEDDING_TYPE}_fine_tune", model_type="FFNN")
        evaluate_model(model, X_test=None, y_test=None,  test_loader=val_loader, device=device, csv_filename=f"{EMBEDDING_TYPE}_fine_tune_eval.csv")

    else:
        print("Using static (non-trainable) embeddings with pooling.")

        for name, pooling_fn in pooling_methods.items():
            print("=" * 60)
            print(f"Testing pooling method: {name}")
            print("=" * 60)

            dim = EMBEDDING_DIM if name not in ["mean_max", "k_max"] else EMBEDDING_DIM * 2
            X, y_vec = create_dataset(abstracts, y, embeddings, EMBEDDING_DIM, pooling_fn)
            X_train, X_dev, y_train, y_dev = train_test_split(X, y_vec, test_size=0.2, stratify=y_vec, random_state=42)

            train_tensor = torch.tensor(X_train, dtype=torch.float32)
            train_labels = torch.tensor(y_train, dtype=torch.long)
            dev_tensor = torch.tensor(X_dev, dtype=torch.float32)
            dev_labels = torch.tensor(y_dev, dtype=torch.long)

            train_loader = [(train_tensor, train_labels)]
            val_loader = [(dev_tensor, dev_labels)]

            model = FFNN(
                input_dim=dim,
                hidden_dims=[256, 128, 64],
                output_dim=len(label_to_idx)
            )

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            idx_to_label = {idx: label for label, idx in label_to_idx.items()}

            train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, device=device, name=f"{EMBEDDING_TYPE}_{name}", model_type="FFNN")
            evaluate_model(model, X_test=dev_tensor, y_test=dev_labels, device=device, csv_filename=f"{EMBEDDING_TYPE}_{name}_eval.csv", idx_to_label=idx_to_label)
