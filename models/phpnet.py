import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GatedGraphConv

class PhpNetGraphTokensCombine(nn.Module):
    def __init__(self):
        super(PhpNetGraphTokensCombine, self).__init__()
        # Embedding layer for tokens
        self.embed1 = nn.Embedding(num_embeddings=50141, embedding_dim=200)

        # Gated Graph Neural Network (GGNN)
        self.ggnn = GatedGraphConv(in_feats=200, out_feats=6000, n_steps=3, n_etypes=6)

        # GRU for sequence modeling
        self.lstm1 = nn.GRU(input_size=200, hidden_size=6000, num_layers=3, batch_first=True, bidirectional=True)

        # Linear layers to reduce feature sizes
        self.fc_graph = nn.Linear(6000, 3000)  # Reduce graph feature size
        self.fc_tokens = nn.Linear(6000, 3000)  # Update to expect the correct reduced token feature size

        # Fully connected layers for classification
        self.lin1 = nn.Linear(6000, 1000)  # Concatenated reduced graph and token features
        self.lin11 = nn.Linear(1000, 500)
        self.lin2 = nn.Linear(500, 2)  # Binary classification output

    def forward(self, dataTokens=None, dataGraph=None):
        x_graph = None  # Initialize graph features
        x_tokens = None  # Initialize token features

        # Process graph data
        if dataGraph is not None:
            features = dataGraph.ndata['type']
            edge_types = dataGraph.edata['label']
            x_graph = self.ggnn(dataGraph, features, edge_types)

            # Debugging print for GGNN output shape
            print("GGNN output shape (before pooling):", x_graph.shape)

            # Use max pooling across nodes to get a richer feature representation
            x_graph = dgl.max_nodes(dataGraph, feat='type')

            # Debugging print for graph shape after pooling
            print("GGNN output shape (after pooling):", x_graph.shape)

            # Ensure the pooled graph output has the correct feature dimension
            x_graph = F.relu(self.fc_graph(x_graph))

        # Process token data
        if dataTokens is not None:
            x_tokens = self.embed1(dataTokens)
            x_tokens, _ = self.lstm1(x_tokens)  # x_tokens shape: [batch_size, seq_length, hidden_size * 2]

            # Debugging print for LSTM output shape
            print("LSTM output shape:", x_tokens.shape)

            # Pooling over sequence length to reduce the dimension
            x_tokens = torch.mean(x_tokens, dim=1)  # Shape: [batch_size, hidden_size * 2]

            # Debugging print for pooled token shape
            print("Pooled token shape:", x_tokens.shape)

            # Reduce token feature size
            x_tokens = F.relu(self.fc_tokens(x_tokens))

        # Concatenate graph and token features if both are available
        if x_graph is not None and x_tokens is not None:
            x = torch.cat((x_graph, x_tokens), dim=1)
        elif x_graph is not None:
            x = x_graph
        else:
            x = x_tokens

        # Pass through fully connected layers
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin11(x))
        x = self.lin2(x)

        return x
