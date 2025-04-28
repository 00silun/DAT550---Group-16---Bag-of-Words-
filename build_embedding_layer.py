import torch
import torch.nn as nn
import numpy as np

def buildEmbeddingLayer(embeddings_dict, embedding_dim, freeze=False):
    print("Building embedding layer...")
    vocab = list(embeddings_dict.keys())
    word2idx = {word: i+2 for i, word in enumerate(vocab)}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1

    embedding_matrix = np.zeros((len(word2idx), embedding_dim))
    for word, idx in word2idx.items():
        if word in embeddings_dict:
            embedding_matrix[idx] = embeddings_dict[word]
        elif word == '<UNK>':
            embedding_matrix[idx] = np.mean(list(embeddings_dict.values()), axis=0)

    embedding_tensor = torch.FloatTensor(embedding_matrix)
    embedding_layer = nn.Embedding.from_pretrained(embedding_tensor, freeze=freeze)

    return embedding_layer, word2idx
