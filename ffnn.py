import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNN(nn.Module):
    def __init__(self, input_dim=None, embedding_layer=None, hidden_dims=[512, 256], output_dim=10, activation='relu', dropout_prob=0.3):
        super().__init__()

        if embedding_layer is not None:
            self.use_embedding = True
            self.embedding = embedding_layer
            input_dim = embedding_layer.embedding_dim  # override input_dim
        else:
            self.use_embedding = False

        self.activation_name = activation
        self.dropout_prob = dropout_prob

        # Build hidden layers dynamically
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Dropout(dropout_prob))

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        if self.use_embedding:
            x = self.embedding(x)  # (batch, seq_len, embedding_dim)
            x = torch.sum(x, dim=1)  # Pooling over sequence dimension

        activations = {
            'relu': F.relu,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'leaky_relu': lambda x: F.leaky_relu(x, 0.01),
            'elu': lambda x: F.elu(x, alpha=1.0)
        }
        act = activations.get(self.activation_name, F.relu)

        for i in range(0, len(self.hidden_layers), 2):
            x = act(self.hidden_layers[i](x))
            x = self.hidden_layers[i + 1](x)

        x = self.output_layer(x)
        return x
