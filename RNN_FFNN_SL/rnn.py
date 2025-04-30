import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=10,
        rnn_type='gru',
        bidirectional=False,
        dropout_prob=0.3,
        padding_idx=0,
        pooling='last',
        pretrained_embeddings=None,
        freeze_embeddings=False,
        num_layers=3 
    ):
        super().__init__()
        self.pooling = pooling
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        rnn_class = {
            'gru': nn.GRU,
            'lstm': nn.LSTM,
            'rnn': nn.RNN
        }.get(self.rnn_type, nn.GRU)

        self.rnn_layers = nn.ModuleList()
        input_size = embedding_dim
        self.layer_dims = []

        for i in range(num_layers):
            out_dim = max(1, hidden_dim // (2 ** i))  # Prevent hidden_dim=0
            self.rnn_layers.append(
                rnn_class(
                    input_size=input_size,
                    hidden_size=out_dim,
                    batch_first=True,
                    bidirectional=bidirectional
                )
            )
            self.layer_dims.append(out_dim)
            input_size = out_dim * (2 if bidirectional else 1)

        rnn_output_dim = input_size
        self.fc = nn.Linear(rnn_output_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        out = x

        for rnn in self.rnn_layers:
            if self.rnn_type == 'lstm':
                out, (hidden, _) = rnn(out)
            else:
                out, hidden = rnn(out)

        # Pooling
        if self.pooling == 'last':
            if isinstance(hidden, tuple):  
                hidden = hidden[0]
            if hidden.size(0) == 2:  
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                hidden = hidden[-1]
            out = self.fc(hidden)

        elif self.pooling == 'max':
            out, _ = torch.max(out, dim=1)
            out = self.fc(out)

        elif self.pooling == 'average':
            out = torch.mean(out, dim=1)
            out = self.fc(out)

        elif self.pooling == 'sum':
            out = torch.sum(out, dim=1)
            out = self.fc(out)

        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

        return out
