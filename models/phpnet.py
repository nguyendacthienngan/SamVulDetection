import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GatedGraphConv

class PhpNetGraphTokensCombine(nn.Module):
    def __init__(self):
        super(PhpNetGraphTokensCombine, self).__init__()
        # Update the embedding size to cover the maximum token ID range
        self.embed1 = nn.Embedding(num_embeddings=50141, embedding_dim=200)  # Updated size

        # Gated Graph Neural Network (GGNN) with adjusted feature sizes
        self.ggnn = GatedGraphConv(in_feats=200, out_feats=6000, n_steps=3, n_etypes=6)  # Updated feature sizes

        # GRU for sequence modeling (optional for tokens)
        self.lstm1 = nn.GRU(input_size=200, hidden_size=6000, num_layers=3, batch_first=True, bidirectional=True)

        # Linear layers to reduce graph and token feature sizes before concatenation
        self.fc_graph = nn.Linear(6000, 3000)  # Reduce graph feature size
        self.fc_tokens = nn.Linear(36000, 3000)  # Update to match GRU output size

        # Fully connected layers
        self.lin1 = nn.Linear(6000, 1000)  # Concatenated reduced graph and token features
        self.lin11 = nn.Linear(1000, 500)
        self.lin2 = nn.Linear(500, 2)  # Output 2 logits for binary classification
    
    def forward(self, dataTokens=None, dataGraph=None):
        x_graph = None  # Initialize x_graph
        x_tokens = None  # Initialize x_tokens
        
        # Process graph data
        if dataGraph is not None:
            features = dataGraph.ndata['type']
            edge_types = dataGraph.edata['label']
            x_graph = self.ggnn(dataGraph, features, edge_types)
            print("GGNN output shape (before pooling):", x_graph.shape)  # Debugging print

            # Pooling (mean pooling)
            # x_graph = dgl.mean_nodes(dataGraph, 'type')  # Example pooling method
            # print("GGNN output shape (after pooling):", x_graph.shape)  # Debugging print
            
            # Ensure x_graph has the correct shape
            if x_graph.dim() == 1:  # If it returns [batch_size], make sure to reshape it
                x_graph = x_graph.unsqueeze(1)  # Change shape to [batch_size, 1]
            
            # Reduce graph feature size
            x_graph = F.relu(self.fc_graph(x_graph))
        
        # Process token data
        if dataTokens is not None:
            x_tokens = self.embed1(dataTokens)
            x_tokens, _ = self.lstm1(x_tokens)
            batch_size = x_tokens.size(0)
            x_tokens = x_tokens.contiguous().view(batch_size, -1)
            x_tokens = F.relu(self.fc_tokens(x_tokens))
        
        # Concatenate features if both are available
        if x_graph is not None and x_tokens is not None:
            x = torch.cat((x_graph, x_tokens), dim=1)
        elif x_graph is not None:
            x = x_graph  # Only graph features are present
        else:
            x = x_tokens  # Only token features are present

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin11(x))
        x = self.lin2(x)

        return x
