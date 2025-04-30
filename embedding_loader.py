from gensim.models import KeyedVectors
from gensim.models import Word2Vec, FastText
from gensim.models.fasttext import load_facebook_vectors
from tqdm import tqdm
import numpy as np

class EmbeddingLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_fasttext(path):
        print("Loading FastText vectors...")
        model = load_facebook_vectors(path)
        return {word: model[word] for word in model.index_to_key}
    
    def load_custom_fasttext(path):
        print("Loading custom Gensim FastText model...")
        model = FastText.load(path) 
        return {word: model.wv[word] for word in model.wv.index_to_key}

    @staticmethod
    def load_word2vec(path, binary=True):
        print("Loading Word2Vec model...")
        model = KeyedVectors.load_word2vec_format(path, binary=binary)
        return {word: model[word] for word in model.index_to_key}

    @staticmethod
    def load_glove(path):
        print("Loading GloVe embeddings...")
        embeddings = {}
        with open(path, encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading GloVe"):
                parts = line.strip().split()
                word = parts[0]
                vector = np.asarray(parts[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings
