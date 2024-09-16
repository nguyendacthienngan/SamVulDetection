import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_max_pool  # Using GatedGraphConv
class PhpNetGraphTokensCombine(nn.Module):
    def __init__(self):
        super(PhpNetGraphTokensCombine, self).__init__()
        # Embedding for graph node features
        self.embed1 = nn.Embedding(num_embeddings=5000, embedding_dim=100)
        
        # Gated Graph Neural Network (GGNN)
        self.ggnn = GatedGraphConv(out_channels=2000, num_layers=3)
        
        # Embedding for token inputs
        self.embed = nn.Embedding(num_embeddings=50141, embedding_dim=100)
        
        # GRU for sequence modeling
        self.lstm1 = nn.GRU(input_size=100, hidden_size=200, num_layers=3, batch_first=True, bidirectional=True)

        # Fully connected layers
        self.lin1 = nn.Linear(3200, 1000)  # Adjust input dimension to 3200
        self.lin11 = nn.Linear(1000, 500)
        self.lin2 = nn.Linear(500, 2)  # 2 classes for binary classification

    def forward(self, dataTokens=None, dataGraph=None):
        if dataGraph is not None:
            features = dataGraph.ndata['type'].long()
            x = self.embed1(features).reshape(features.shape[0], -1)
            x = self.ggnn(x, dataGraph.edge_index)
            batch = dataGraph.batch if hasattr(dataGraph, 'batch') else None
            x = global_max_pool(x, batch)
        else:
            batch_size = dataTokens.size(0)
            x = torch.zeros(batch_size, 2000).to(dataTokens.device)

        x1 = self.embed(dataTokens)
        output1, hidden1 = self.lstm1(x1)
        x1 = torch.cat((hidden1[0, :, :], hidden1[1, :, :], hidden1[2, :, :], hidden1[-3, :, :], hidden1[-2, :, :], hidden1[-1, :, :]), dim=1)

        x = torch.cat([x, x1], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin11(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin2(x))

        return x
