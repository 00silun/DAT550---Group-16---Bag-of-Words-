# Bag of Words Document Classification - DAT550 - Group 16


## 1. Project Overview
This project classifies scientific abstracts into 10 research fields (Computer Science, Physics, Mathematics, etc.) using neural networks.  
We work with the **Arxiv-10** dataset.

**Goal:** Predict the research field based on the abstract


## 2. Requirements

- Python 3.10+

## 3. Project Structure

| Folder / File | Purpose |
|---------------|---------|
| `RNN_FNN_main.py` | **Main file** for running experiments (RNN and FFNN). |
| `staticEmbeddingClassifier.py` | Older experiment file (uses static embedding averaging). |
| `prepareData.py` | Prepares datasets and data loaders. |
| `trainModel.py` | Training logic (loop over epochs, optimizer, loss computation). |
| `evaluation.py` | Evaluation functions (accuracy, precision, recall, F1, confusion matrix). |
| `ffnn.py` | Defines the FFNN model architecture. |
| `rnn.py` | Defines the RNN model architecture (GRU, LSTM, or simple RNN). |
| `load_embeddings.py` | Loads pre-trained embeddings like GloVe. |
| `cleanText.py` | Cleans text abstracts (lowercasing, punctuation removal). |

## 4. Steps to Reproduce Results

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

4.  **Prepare Embeddings and Dataset**:

    Download the following files and place them into the specified subfolders within the project directory:

    * **Embeddings** (Required in `embeddings/` folder):
        * **Word2Vec:** `GoogleNews-vectors-negative300.bin`
            * Source: [GoogleNews Word2Vec 300d](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g)
        * **FastText:** `crawl-300d-2M-subword.bin`
            * Source: [FastText Crawl 300d](https://fasttext.cc/docs/en/english-vectors.html)
        * **GloVe:** `glove.6B.300d.txt`
            * Source: [GloVe 6B 300d](https://nlp.stanford.edu/projects/glove/)

    * **Dataset** (Required in `data/` folder):
        * **arXiv100:** `arxiv100.csv`
            * Source: [arXiv100 Sample Dataset](https://liveuis-my.sharepoint.com/personal/2926110_uis_no/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F2926110%5Fuis%5Fno%2FDocuments%2FAttachments%2FArchive%2Ezip&parent=%2Fpersonal%2F2926110%5Fuis%5Fno%2FDocuments%2FAttachments&ga=1)

5. **Configure settings**:
   - Inside `RNN_FNN_main.py`, update the `config` dictionary if needed

6. **Run the model**:

    ```bash
    python RNN_FNN_main.py
    ```


## 5. Outputs

- After training:
  - Best model weights are saved internally.
  - Evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix) are saved as CSV files in the `outputs/` directory.
  - Model predictions and confusion matrix plots can be generated for analysis.


## 6. Notes

- `RNN_FNN_main.py` is the **primary file** 
- Hyperparameters like `dropout`, `learning_rate`, and `embedding_type` are easily adjustable in the `config` block inside the main file.
- We recomend running this on a powerfull computer or VM

