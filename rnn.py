import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=64,
        output_dim=10,
        rnn_type='gru',
        bidirectional=False,
        dropout_prob=0.3,
        padding_idx=0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # Choose RNN type
        rnn_class = {
            'gru': nn.GRU,
            'lstm': nn.LSTM,
            'rnn': nn.RNN
        }.get(rnn_type.lower(), nn.GRU)

        self.rnn = rnn_class(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )

        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(rnn_output_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.rnn_type = rnn_type

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        if self.rnn_type.lower() == 'lstm':
            _, (hidden, _) = self.rnn(embedded)
        else:
            _, hidden = self.rnn(embedded)

        # If bidirectional, concatenate last forward and backward hidden states
        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[0]
        if hidden.size(0) == 2:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        out = self.fc(hidden)
        return out
