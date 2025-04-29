import numpy as np
import torch
from gensim.models import KeyedVectors, FastText
import os

def load_pretrained_embeddings(vocab, embedding_type="glove"):
    """
    Loads pre-trained embeddings and returns a torch tensor and its dimension.

    Args:
        vocab (dict): word -> index mapping
        embedding_type (str): 'glove', 'word2vec', or 'fasttext'

    Returns:
        (torch.FloatTensor, int): Embedding matrix and embedding dimension
    """
    # Set default paths and dimensions
    if embedding_type == "glove":
        path = "../embeddings/glove.6B.100d.txt"
        embedding_dim = 100
    elif embedding_type == "word2vec":
        path = "../embeddings/GoogleNews-vectors-negative300.bin"
        embedding_dim = 300
    elif embedding_type == "fasttext":
        path = "../embeddings/crawl-300d-2M-subword.bin"
        embedding_dim = 300
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

    print(f"Loading {embedding_type} embeddings from {path}...")

    embeddings_index = {}

    if embedding_type == "glove":
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = vector

    elif embedding_type == "word2vec":
        model = KeyedVectors.load_word2vec_format(path, binary=True)
        for word in vocab:
            if word in model:
                embeddings_index[word] = model[word]

    elif embedding_type == "fasttext":
        model = FastText.load_fasttext_format(path)
        for word in vocab:
            if word in model.wv:
                embeddings_index[word] = model.wv[word]

    print(f"Found {len(embeddings_index)} word vectors.")

    # Build embedding matrix
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in vocab.items():
        vector = embeddings_index.get(word)
        if vector is not None and len(vector) == embedding_dim:
            embedding_matrix[idx] = vector
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    return torch.tensor(embedding_matrix, dtype=torch.float32), embedding_dim
