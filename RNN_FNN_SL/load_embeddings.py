# load_embeddings.py

import numpy as np
import torch

def load_pretrained_embeddings(vocab, embedding_dim=100, embedding_path="/zfs1/home/u256437/DAT550Project/glove.6B.100d.txt"
):
    """
    Loads pre-trained GloVe embeddings and matches them with the vocab.

    Args:
        vocab (dict): word -> index mapping
        embedding_dim (int): Dimension of embeddings
        embedding_path (str): Path to pre-trained GloVe file

    Returns:
        torch.FloatTensor: Embedding matrix
    """
    print("Loading pre-trained embeddings...")

    # 1. Load embeddings
    embeddings_index = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    print(f"Found {len(embeddings_index)} word vectors in GloVe file.")

    # 2. Create embedding matrix
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, idx in vocab.items():
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))  # random for unknown words

    # 3. Convert to tensor
    embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)

    return embedding_tensor
