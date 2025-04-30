# Bag of Words Document Classification - DAT550 - Group 16

## 1. Project Overview

This project classifies scientific abstracts into 10 research fields (Computer Science, Physics, Mathematics, etc.) using neural networks. We work with the **Arxiv-10** dataset.

## 2. Requirements

- Python 3.10+

## 3. Steps to Reproduce Results

1. **Clone the repository**:

   - HTTPS:

     ```bash
     git clone https://github.com/00silun/DAT550---Group-16---Bag-of-Words-.git
     cd DAT550---Group-16---Bag-of-Words-
     ```

   - SSH:
     ```bash
     git clone git@github.com:00silun/DAT550---Group-16---Bag-of-Words-.git
     cd DAT550---Group-16---Bag-of-Words-
     ```

2. **Set up a Python virtual environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install requirements**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Embeddings and Dataset**:

   Download the following files and place them into the specified subfolders within the project directory:

- **Embeddings** (Required in `embeddings/` folder):

  - **Word2Vec:** `GoogleNews-vectors-negative300.bin`
    - Source: [GoogleNews Word2Vec 300d](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g)
  - **FastText:** `crawl-300d-2M-subword.bin`
    - Source: [FastText Crawl 300d](https://fasttext.cc/docs/en/english-vectors.html)
  - **GloVe:** Download the GloVe 6B embeddings and ensure you have the 100, 200, and 300-dimensional files:

    - `glove.6B.100d.txt`
    - `glove.6B.200d.txt`
    - `glove.6B.300d.txt`

    - Source: [glove.6B.zip](https://nlp.stanford.edu/projects/glove/)

- **Dataset** (Required in `data/` folder):
  - **arXiv100:** `arxiv100.csv`
    - Source: [arXiv100 Sample Dataset](https://paperswithcode.com/dataset/arxiv-10)

1. **Run the models**:

   **Running Feedforward Neural Networks (FFNN) with static embeddings:**

   ```bash
   python staticEmbeddingClassifier.py
   ```

   - This script trains FFNN models.
   - Supports static embeddings (GloVe, Word2Vec, FastText).
   - Also supports custom-trained embeddings and fine-tuning.

   **Running Recurrent Neural Networks (RNN) or FFNN with Bag-of-Words:**

   ```bash
   cd RNN_FFNN_SL
   python RNN_FNN_main.py
   ```

   - This script trains both RNN (GRU, LSTM) and FFNN models.
   - It supports:
     - Bag-of-Words (CountVectorizer, TF-IDF)
     - Word Embeddings (GloVe, Word2Vec, FastText)
   - Configure config inside the script:
     - model_type, rnn_type, embedding_type, pooling
     - Hyperparameters (batch size, learning rate, epochs)

## 4. Outputs

- After training:
  - The model saves the best weights automatically based on validation loss.
  - Evaluation metrics (accuracy, precision, recall, F1-score) are saved as CSV files.
  - Confusion matrix plots can be generated using `plotResult.py`.

## 5. Notes

- We recomend running this on a powerfull computer or VM
