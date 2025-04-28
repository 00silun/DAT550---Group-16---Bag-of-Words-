import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import nltk
nltk.download('punkt')

from prepareData import prepare_data
from ffnn import FFNN
from rnn import RNN
from evaluation import evaluate_model
from trainModel import train_model
from cleanText import clean_text
from load_embeddings import load_pretrained_embeddings
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------
# CONFIGURATION
# ------------------------------
config = {
    # Model selection
    "model_type": "RNN",        # Options: 'FFNN' (bag-of-words) or 'RNN' (sequence model)

    # RNN-specific options
    "rnn_type": "gru",          # Options (only if model_type='RNN'): 'rnn', 'lstm', 'gru'
    "pooling": "max",        # Options (only if model_type='RNN'): 'last', 'average', 'max'

    # Embedding options
    "embedding_type": "pretrained",  # Options (only if model_type='RNN'): 'random' (learn from scratch), 'pretrained' (e.g., GloVe)
    "freeze_embeddings": False,      # True = Freeze pre-trained embeddings, False = Fine-tune during training

    # FFNN-specific options
    "use_tfidf": True,           # Only used if model_type='FFNN'. Options: True (use TF-IDF) or False (use CountVectorizer)

    # Data and input size
    "vocab_size": 20000,          # Size of vocabulary (for RNN) or max number of features (for FFNN)
    "embedding_dim": 100,         # Dimensionality of word embeddings (e.g., 50, 100, 200, 300 for GloVe)
    "max_seq_len": 300,           # Maximum number of tokens (words) per abstract

    # Training hyperparameters
    "batch_size": 64,             # Number of samples per batch
    "num_epochs": 14,             # Number of full passes through the training data
    "activation": "relu",         # Activation function to use (mainly for FFNN): 'relu', 'tanh', etc.
    "dropout_prob": 0.4,          # Dropout probability (helps prevent overfitting)
    "learning_rate": 0.00035,     # Learning rate for optimizer (lower = slower but more stable)

    # Dataset path
    "dataset_path": "/zfs1/home/u256437/DAT550Project/arxiv100.csv"  # Path to CSV file
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create test_loader here (NEW)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=config["batch_size"], shuffle=False)

# ------------------------------
# MODEL SETUP
# ------------------------------
print(f"Initializing {config['model_type']} model...")
if config["model_type"] == "FFNN":
    model = FFNN(
        input_dim=input_dim,
        hidden_dims=[1024, 512, 256],
        output_dim=output_dim,
        activation=config["activation"],
        dropout_prob=config["dropout_prob"]
    )
else:  # RNN
    if config["embedding_type"] == "pretrained":
        pretrained_embeddings = load_pretrained_embeddings(
            vocab,
            embedding_dim=config["embedding_dim"],
            embedding_path="/zfs1/home/u256437/DAT550Project/glove.6B.100d.txt"
        )
    else:
        pretrained_embeddings = None

    model = RNN(
        vocab_size=input_dim,
        embedding_dim=config["embedding_dim"],
        hidden_dim=128,
        output_dim=output_dim,
        rnn_type=config["rnn_type"],
        pooling=config["pooling"],
        bidirectional=True,
        dropout_prob=config["dropout_prob"],
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=config["freeze_embeddings"]
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
print("Evaluating model...")
results = evaluate_model(
    model,
    test_loader,  # <-- pass DataLoader, not raw tensors
    device=device,
    csv_filename=f'{config["model_type"].lower()}_evaluation_log.csv'
)

print("Final Evaluation Metrics:", results)