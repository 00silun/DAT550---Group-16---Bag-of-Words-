import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from evaluation import evaluate_model
from trainModel import train_model
from cleanText import clean_text
from ffnn import FFNN
from rnn import RNN
from smartDataPreparatio import prepare_data  # <- import smart data loader

# ------------------------------
# Config
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = 'FFNN'  # RNN or 'FFNN'
vocab_size = 10000
max_seq_len = 200
batch_size = 64
activation = 'relu'
num_epochs = 5
use_tfidf=True
# ------------------------------
# Load & Preprocess
# ------------------------------
print("Loading and preprocessing data...")
df = pd.read_csv("arxiv100.csv")
df['processed_abstract'] = df['abstract'].apply(clean_text)

train_loader, val_loader, X_test_tensor, y_test_tensor, input_dim, output_dim, vocab = prepare_data(
    df,
    model_type=model_type,
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    batch_size=batch_size,
    use_tfidf=use_tfidf
)

# ------------------------------
# Initialize Model
# ------------------------------
print(f"Initializing {model_type} model...")
if model_type == 'FFNN':
    model = FFNN(
        input_dim=input_dim,
        hidden_dims=[512, 256],
        output_dim=output_dim,
        activation=activation,
        dropout_prob=0.3
    )
else:
    model = RNN(
        vocab_size=input_dim,
        embedding_dim=128,
        hidden_dim=64,
        output_dim=output_dim,
        rnn_type='lstm',
        bidirectional=True,
        dropout_prob=0.4
    )
model.to(device)

# ------------------------------
# Train
# ------------------------------
print("Training model...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

train_losses, val_losses, best_model_state = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs,
    device=device,
    name=model_type,
    model_type=model_type
)

model.load_state_dict(best_model_state)

# ------------------------------
# Evaluate
# ------------------------------
print("Evaluating model...")
results = evaluate_model(
    model, X_test_tensor, y_test_tensor,
    device=device,
    csv_filename=f'{model_type.lower()}_evaluation_log.csv'
)
print("Final Evaluation Metrics:", results)
