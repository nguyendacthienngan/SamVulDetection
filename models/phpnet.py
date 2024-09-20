import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

class ModelParser():
    vector_size: int = 100  # 图结点的向量维度
    hidden_size: int = 256  # GNN隐层向量维度
    layer_num: int = 5  # GNN层数
    num_classes: int = 2
model_args = ModelParser()

class PhpNetGraphTokensCombine(nn.Module):
    def __init__(self):
        super(PhpNetGraphTokensCombine, self).__init__()
        # Example LSTM layer; adjust parameters as needed
        embedding_dim=768
        vocab_size=50265
        max_edge_types=6
        input_dim = max_edge_types + model_args.vector_size
        output_dim = model_args.hidden_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_edge_types = max_edge_types
        # Embedding layer for tokens
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # Gated Graph Neural Network (GGNN)
        self.ggnn = GatedGraphConv(in_feats=200, out_feats=6000, n_steps=3, n_etypes=6)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=12000, num_layers=1, batch_first=True)
        
        # Fully connected layers for token features
        self.fc_tokens = nn.Linear(12000, 3000)  # Adjust the output dimension if necessary
        
        # Example GNN layers; adjust parameters as needed
        self.fc_graph = nn.Linear(6000, 3000)  # Change this based on your actual output from GNN

        # Output layer for final classification
        # Concatenated features (3000 from tokens + 3000 from graph = 6000)
        self.output_layer = nn.Linear(6000, 2)  # Output for binary classification
    
    def forward(self, dataTokens, dataGraph=None):
        # Ensure dataTokens is correctly shaped: [batch_size, seq_len]
        # Convert token IDs to embeddings if necessary
        if dataTokens.ndim == 2:  # Check if dataTokens is [batch_size, seq_len]
            dataTokens = self.embedding(dataTokens)  # Assuming you have an embedding layer

        # Process token data through LSTM
        lstm_out, _ = self.lstm(dataTokens)  # Check that input shape is [batch_size, seq_len, embedding_dim]
        
        # Pool LSTM output
        x_tokens = lstm_out.mean(dim=1)  # Average pooling over the sequence length
        print("Pooled token shape:", x_tokens.shape)  # Should be [batch_size, 12000]
        
        # Reduce token feature size
        x_tokens = F.relu(self.fc_tokens(x_tokens))
        
        # Process graph data
        if dataGraph is not None:
            features = dataGraph.ndata['type']  # Use 'type' as the feature input
            edge_types = dataGraph.edata['label']  # Use 'label' as edge type

            outputs = self.ggnn(dataGraph, features, edge_types)
            dataGraph.ndata['GGNNOUTPUT'] = outputs
            x_i, h_i = self.process_graph(dataGraph)  # Process the graph
            
            # Pooling for graph data
            if isinstance(h_i, list):  # Check if h_i is a list of tensors
                x_graph = torch.stack([torch.mean(h, dim=0) for h in h_i])  # Process each tensor in the list
            else:  # h_i is already a single tensor
                x_graph = torch.mean(h_i, dim=0)


            # Ensure the dimension matches the expected input to the linear layer
            if x_graph.shape[-1] != 6000:  # Check if the last dimension is 6000
                # If not, you can either reshape or ensure the GGNN outputs the correct shape.
                # Example fix: assuming you want to flatten x_graph or apply further processing:
                print('x_graph.shape[-1] != 6000')
                x_graph = torch.flatten(x_graph, start_dim=1)  # Flatten for fully connected layers
                # Alternatively, you can add another linear layer here to map to the correct size

            x_graph = F.relu(self.fc_graph(x_graph))  # Now pass it through the linear layer

            # Concatenate graph and token features
            combined_features = torch.cat((x_tokens, x_graph), dim=1)
        else:
            # If no graph input, concatenate token features with a zero tensor to keep size consistent
            zero_graph = torch.zeros_like(x_tokens)  # Create a zero tensor of same shape as x_tokens
            combined_features = torch.cat((x_tokens, zero_graph), dim=1)  # Still (batch_size, 6000)        
        
        # Final output layer (define this layer in __init__)
        output = self.output_layer(combined_features)  # Adjust based on your task
        
        return output

    def process_graph(self, g):
        x_i = []
        h_i = []
        max_len = -1
        for g_i in dgl.unbatch(g):
            x_i.append(g_i.ndata['type'])
            h_i.append(g_i.ndata['GGNNOUTPUT'])
            max_len = max(g_i.number_of_nodes(), max_len)
        for i, (v, k) in enumerate(zip(x_i, h_i)):
            if v.size(1) != self.input_dim:
                v = F.pad(v, (0, self.input_dim - v.size(1)), value=0)
            if k.size(1) != self.output_dim:
                k = F.pad(k, (0, self.output_dim - k.size(1)), value=0)

            x_i[i] = torch.cat(
                (v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad,
                                device=v.device)), dim=0)
            h_i[i] = torch.cat(
                (k, torch.zeros(size=(max_len - k.size(0), *(k.shape[1:])), requires_grad=k.requires_grad,
                                device=k.device)), dim=0)
        return x_i, h_i
