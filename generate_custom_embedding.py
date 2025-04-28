import os
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess

def generateCustomEmbeddings(corpus_csv_path, corpus_txt_path, save_dir, dim=300, min_count=3, window=5, epochs=10):
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
