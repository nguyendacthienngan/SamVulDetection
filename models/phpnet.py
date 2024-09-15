import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, EdgePooling
from torch_geometric.nn import global_max_pool

class PhpNetGraphTokensCombine(nn.Module):
    def __init__(self):
        super(PhpNetGraphTokensCombine, self).__init__()
        self.embed1 = nn.Embedding(num_embeddings=5000, embedding_dim=100)
        self.conv1 = GCNConv(2000, 2000)
        self.pool1 = EdgePooling(2000)
        self.conv2 = GCNConv(2000, 4000)
        self.pool2 = EdgePooling(4000)
        self.conv3 = GCNConv(4000, 4000)
        self.pool3 = EdgePooling(4000)
        self.conv4 = GCNConv(2000, 2000)
        self.pool4 = EdgePooling(8000)

        self.embed = nn.Embedding(num_embeddings=5000, embedding_dim=100)
        self.lstm1 = nn.GRU(input_size=100, hidden_size=200, num_layers=3, batch_first=True, bidirectional=True)

        self.lin1 = nn.Linear(5200, 1000)
        self.lin11 = nn.Linear(1000, 500)
        self.lin2 = nn.Linear(500, 4)
        self.lin3 = nn.Linear(200, 100)
        self.lin4 = nn.Linear(100, 4)

    def forward(self, dataTokens=None, dataGraph=None):
        # Check if graph inputs are provided
        if dataGraph is not None:
            # Updated graph data access similar to DevignModel
            features = dataGraph.ndata['type'].long()  # Assuming 'type' contains node features
            edge_types = dataGraph.edata['label'].long()  # Assuming 'label' contains edge types
            batch = dataGraph.batch

            # Perform graph convolution and pooling
            x = self.embed1(features).reshape(features.shape[0], -1)
            x = self.conv1(x, dataGraph.edge_index)
            x, edge_index, batch, _ = self.pool1(x, dataGraph.edge_index, batch=batch)
            x = F.relu(x)

            x = self.conv2(x, edge_index)
            x, edge_index, batch, _ = self.pool2(x, edge_index, batch=batch)
            x = F.relu(x)

            x = self.conv3(x, edge_index)
            x, edge_index, batch, _ = self.pool3(x, edge_index, batch=batch)
            x = F.relu(x)

            # Global pooling of graph features
            x = global_max_pool(x, batch)
        else:
            # If no graph input is provided, use a zero tensor for `x`
            batch_size = dataTokens.size(0)  # Get batch size from token inputs
            x = torch.zeros(batch_size, 2000).to(dataTokens.device)  # Assuming 2000 features for graph

        # Process token inputs using GRU
        x1 = self.embed(dataTokens)
        output1, hidden1 = self.lstm1(x1)
        x1 = torch.cat((hidden1[0, :, :], hidden1[1, :, :], hidden1[2, :, :], hidden1[-3, :, :], hidden1[-2, :, :], hidden1[-1, :, :]), dim=1)

        # Concatenate graph and token features
        x = torch.cat([x, x1], dim=1)

        # Fully connected layers
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin11(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin2(x))

        return x
