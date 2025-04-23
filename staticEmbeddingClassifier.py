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
import os
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess

from gensim.models.fasttext import load_facebook_vectors
#nltk.download('stopwords') # Uncomment this line if stopwords are not downloaded

from gensim.models import FastText

def load_custom_fasttext_gensim(path):
    print("Loading custom Gensim FastText model...")
    model = FastText.load(path)  # Load the actual model
    return {word: model.wv[word] for word in model.wv.index_to_key}


def generate_custom_embeddings(corpus_csv_path, corpus_txt_path, save_dir, dim=300, min_count=3, window=5, epochs=10):
    print("\n[Step 1] Extracting corpus from CSV...")
    df = pd.read_csv(corpus_csv_path)
    df['abstract'] = df['abstract'].astype(str).str.strip().str.lower()
    with open(corpus_txt_path, "w", encoding="utf-8") as f:
        for abstract in df['abstract']:
            f.write(abstract + "\n")
    print(f"Saved {len(df)} abstracts to {corpus_txt_path}")

    print("\n[Step 2] Preprocessing corpus...")
    with open(corpus_txt_path, 'r', encoding='utf-8') as f:
        sentences = [simple_preprocess(line.strip()) for line in tqdm(f)]

    os.makedirs(save_dir, exist_ok=True)

    print("\n[Step 3] Training Word2Vec...")
    w2v_model = Word2Vec(sentences=sentences, vector_size=dim, window=window, min_count=min_count, workers=4, epochs=epochs)
    w2v_model.wv.save_word2vec_format(os.path.join(save_dir, "custom_word2vec.vec"), binary=False)

    print("\n[Step 4] Training FastText...")
    ft_model = FastText(sentences=sentences, vector_size=dim, window=window, min_count=min_count, workers=4, epochs=epochs)
    ft_model.save(os.path.join(save_dir, "custom_fasttext.bin"))

    print("\n✅ Custom embeddings saved to:", save_dir)



# Load Embeddings

def load_fasttext_embeddings(path):
    print("Loading FastText vectors...")
    model = load_facebook_vectors(path)
    return {word: model[word] for word in model.index_to_key}

def load_word2vec_embeddings(path, binary=True):
    print("Loading Word2Vec model...")
    model = KeyedVectors.load_word2vec_format(path, binary=binary)
    return {word: model[word] for word in model.index_to_key}

def load_glove_embeddings(path):
    print("Loading GloVe embeddings...")
    embeddings = {}
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading GloVe"):
            parts = line.strip().split()
            word = parts[0]
            vector = np.asarray(parts[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Preprocessing

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower().strip()
    words = re.findall(r'\b\w+\b', text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df['abstract'] = df['abstract'].astype(str).apply(clean_text)
    return df['abstract'].tolist(), df['label'].tolist()

# Pooling strategies

def mean_pooling(vectors, dim):
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

def max_pooling(vectors, dim):
    return np.max(vectors, axis=0) if vectors else np.zeros(dim)

def sum_pooling(vectors, dim):
    return np.sum(vectors, axis=0) if vectors else np.zeros(dim)

def min_pooling(vectors, dim):
    return np.min(vectors, axis=0) if vectors else np.zeros(dim)

def mean_max_concat(vectors, dim):
    if vectors:
        return np.concatenate((np.mean(vectors, axis=0), np.max(vectors, axis=0)))
    else:
        return np.zeros(dim * 2)

def k_max_pooling(vectors, dim, k=2):
    if vectors:
        vectors_arr = np.array(vectors)
        sorted_vecs = np.sort(vectors_arr, axis=0)
        return sorted_vecs[-k:, :].flatten()
    else:
        return np.zeros(dim * k)

def log_sum_exp_pooling(vectors, dim):
    if vectors:
        return np.log(np.sum(np.exp(vectors), axis=0))
    else:
        return np.zeros(dim)

def attention_pooling(vectors, dim, attention_vector):
    if vectors:
        vectors_arr = np.array(vectors)
        scores = np.dot(vectors_arr, attention_vector)
        weights = np.exp(scores) / np.sum(np.exp(scores))
        return np.sum(vectors_arr * weights[:, np.newaxis], axis=0)
    else:
        return np.zeros(dim)

# Vectorize and dataset creation

def vectorize_text(text, embeddings, dim, pooling_fn, **kwargs):
    words = text.split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    return pooling_fn(vectors, dim, **kwargs) if kwargs else pooling_fn(vectors, dim)

def create_dataset(texts, labels, embeddings, dim, pooling_fn, **kwargs):
    X = np.array([vectorize_text(t, embeddings, dim, pooling_fn, **kwargs) for t in texts])
    y = np.array(labels)
    return X, y

# Main setup

if __name__ == "__main__":



    EMBEDDING_DIM = 300
    DATASET_PATH = "arxiv100.csv"
    EMBEDDING_TYPE = "custom_fasttext"  # Choose between "fasttext", "word2vec", "glove", "custom_word2vec", "custom_fasttext"

    embedding_paths = {
        "fasttext": "crawl-300d-2M-subword.bin",
        "word2vec": "GoogleNews-vectors-negative300.bin.gz",
        "glove": "glove.6B.300d.txt",
        "custom_word2vec": "embeddings/custom_word2vec.vec",
        "custom_fasttext": "embeddings/custom_fasttext.bin"
    }

    pooling_methods = {
        "mean": mean_pooling,
        "max": max_pooling,
        "sum": sum_pooling,
        #"min": min_pooling,
        "mean_max": mean_max_concat,
        #"k_max": lambda vecs, dim: k_max_pooling(vecs, dim, k=2),
        #"log_sum_exp": log_sum_exp_pooling,
        #"attention": lambda vecs, dim: attention_pooling(vecs, dim, attention_vector=np.random.rand(dim))
    }

     # Generate custom embeddings
    #generate_custom_embeddings(
    #    corpus_csv_path="arxiv100.csv",
    #    corpus_txt_path="arxiv_corpus.txt",
    #    save_dir="embeddings",
    #    dim=300,
    #    min_count=3,
    #    window=5,
    #    epochs=10
    #)

    # Update embedding paths to include custom ones
    #embedding_paths.update({
    #    "custom_word2vec": "embeddings/custom_word2vec.vec",
    #    "custom_fasttext": "embeddings/custom_fasttext.bin"
    #})

    # Now continue with your existing pipeline, setting EMBEDDING_TYPE to 'custom_word2vec' or 'custom_fasttext'

    # Load dataset and embeddings

    abstracts, labels = load_dataset(DATASET_PATH)
    label_to_idx = {label: i for i, label in enumerate(sorted(set(labels)))}
    y = [label_to_idx[label] for label in labels]

    if EMBEDDING_TYPE == "fasttext":
        embeddings = load_fasttext_embeddings(embedding_paths[EMBEDDING_TYPE])
    elif EMBEDDING_TYPE == "word2vec":
        embeddings = load_word2vec_embeddings(embedding_paths[EMBEDDING_TYPE], binary=True)
    elif EMBEDDING_TYPE == "glove":
        embeddings = load_glove_embeddings(embedding_paths[EMBEDDING_TYPE])
    elif EMBEDDING_TYPE == "fasttext":
        embeddings = load_fasttext_embeddings(embedding_paths[EMBEDDING_TYPE])
    elif EMBEDDING_TYPE == "custom_fasttext":
        embeddings = load_custom_fasttext_gensim(embedding_paths[EMBEDDING_TYPE])
    elif EMBEDDING_TYPE in ["word2vec", "custom_word2vec"]:
        binary_flag = EMBEDDING_TYPE == "word2vec"  # only pretrained is binary
        embeddings = load_word2vec_embeddings(embedding_paths[EMBEDDING_TYPE], binary=binary_flag)
    elif EMBEDDING_TYPE in ["glove", "custom_glove"]:
        embeddings = load_glove_embeddings(embedding_paths[EMBEDDING_TYPE])
    else:
        raise ValueError("Invalid EMBEDDING_TYPE selected.")

 

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

        model = FFNN(input_dim=dim, hidden_dims=[512, 256,128], output_dim=len(label_to_idx))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), name=f"{EMBEDDING_TYPE}_{name}", model_type="FFNN")

        evaluate_model(model, X_test=dev_tensor, y_test=dev_labels, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), csv_filename=f"{EMBEDDING_TYPE}_{name}_eval.csv")
