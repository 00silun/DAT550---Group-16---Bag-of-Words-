import numpy as np

class Pooling:
    @staticmethod
    def mean(vectors, dim):
        return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

    @staticmethod
    def max(vectors, dim):
        return np.max(vectors, axis=0) if vectors else np.zeros(dim)

    @staticmethod
    def sum(vectors, dim):
        return np.sum(vectors, axis=0) if vectors else np.zeros(dim)

    @staticmethod
    def min(vectors, dim):
        return np.min(vectors, axis=0) if vectors else np.zeros(dim)

    @staticmethod
    def mean_max(vectors, dim):
        if vectors:
            return np.concatenate((np.mean(vectors, axis=0), np.max(vectors, axis=0)))
        else:
            return np.zeros(dim * 2)

    @staticmethod
    def k_max(vectors, dim, k=2):
        if vectors:
            vectors_arr = np.array(vectors)
            sorted_vecs = np.sort(vectors_arr, axis=0)
            return sorted_vecs[-k:, :].flatten()
        else:
            return np.zeros(dim * k)

    @staticmethod
    def log_sum_exp(vectors, dim):
        if vectors:
            return np.log(np.sum(np.exp(vectors), axis=0))
        else:
            return np.zeros(dim)

    @staticmethod
    def attention(vectors, dim, attention_vector):
        if vectors:
            vectors_arr = np.array(vectors)
            scores = np.dot(vectors_arr, attention_vector)
            weights = np.exp(scores) / np.sum(np.exp(scores))
            return np.sum(vectors_arr * weights[:, np.newaxis], axis=0)
        else:
            return np.zeros(dim)
