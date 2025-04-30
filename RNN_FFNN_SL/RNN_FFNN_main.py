import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import nltk
nltk.download('punkt')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from prepareData import prepare_data
from ffnn import FFNN
from rnn import RNN
from trainModel import train_model
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cleanText import clean_text
from evaluation import evaluate_model
from load_embeddings import load_pretrained_embeddings
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------
# CONFIGURATION
# ------------------------------
config = {
    # Model selection
    "model_type": "FFNN",

    # RNN-specific options
    "rnn_type": "GRU",
    "pooling": "max",

    # Embedding options
    "embedding_type": "random",  # 'random' or 'pretrained'
    "embedding_source": "glove",  # 'glove', 'word2vec', or 'fasttext'
    "freeze_embeddings": False,

    # FFNN-specific options
    "use_tfidf": True,

    # Data and input size
    "vocab_size": 20000,
    "max_seq_len": 300,
    "hidden_dim": 256,

    # Training hyperparameters
    "batch_size": 16,
    "num_epochs": 10,
    "activation": "relu",
    "dropout_prob": 0.4,
    "learning_rate": 0.00035,

    # RNN layers
    "num_rnn_layers": 3,

    # Dataset path
    "dataset_path": "../data/arxiv100.csv"
}

# ------------------------------
# DATA PREPARATION
# ------------------------------
print("Loading and preprocessing data...")
df = pd.read_csv(config["dataset_path"])
df['processed_abstract'] = df['abstract'].apply(clean_text)

train_loader, val_loader, X_test_tensor, y_test_tensor, input_dim, output_dim, vocab = prepare_data(
    df,
    model_type=config["model_type"],
    vocab_size=config["vocab_size"],
    max_seq_len=config["max_seq_len"],
    batch_size=config["batch_size"],
    use_tfidf=config["use_tfidf"]
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=config["batch_size"], shuffle=False)

# ------------------------------
# MODEL SETUP
# ------------------------------
print(f"Initializing {config['model_type']} model...")

if config["model_type"] == "FFNN":
    model = FFNN(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        output_dim=output_dim,
        activation=config["activation"],
        dropout_prob=config["dropout_prob"]
    )
else:
    if config["embedding_type"] == "pretrained":
        pretrained_embeddings, embedding_dim = load_pretrained_embeddings(
            vocab,
            embedding_type=config["embedding_source"]
        )
    else:
        pretrained_embeddings = None
        embedding_dim = 100  # Default if not using pretrained

    model = RNN(
        vocab_size=input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=config["hidden_dim"],
        output_dim=output_dim,
        rnn_type=config["rnn_type"],
        pooling=config["pooling"],
        bidirectional=True,
        dropout_prob=config["dropout_prob"],
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=config["freeze_embeddings"],
        num_layers=config["num_rnn_layers"]
    )

model = model.to(device)

# ------------------------------
# TRAINING
# ------------------------------
print("Training model...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

train_losses, val_losses, best_model_state = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=config["num_epochs"],
    device=device,
    name=config["model_type"],
    model_type=config["model_type"]
)

model.load_state_dict(best_model_state)

# ------------------------------
# EVALUATION
# ------------------------------
results = evaluate_model(
    model,
    test_loader=test_loader,
    device=device,
    csv_filename=f'{config["model_type"].lower()}_evaluation_log.csv',
)
