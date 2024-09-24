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
        embedding_dim = 768
        vocab_size = 50265
        max_edge_types = 6
        num_steps=6
        
        # Set the correct dimensions
        self.input_dim = max_edge_types + model_args.vector_size
        self.output_dim = model_args.hidden_size  # Ensure this matches what you need
        self.concat_dim = self.input_dim + self.output_dim  # 106 + 256 = 362

        # Embedding layer for tokens
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # Gated Graph Neural Network (GGNN)
        self.ggnn = GatedGraphConv(in_feats=self.input_dim, out_feats=self.output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=12000, num_layers=1, batch_first=True)
        
        # Fully connected layers for token features
        # self.fc_tokens = nn.Linear(in_features=768, out_features=2)
        self.fc_tokens = nn.Linear(in_features=12000, out_features=3000)  # Match the LSTM output size

        # GNN layers
        self.fc_graph = nn.Linear(in_features=256, out_features=3000)

        # Output layer for final classification
        # self.output_layer = nn.Linear(2, 2)  # Outputting 2 logits for binary classification
        self.output_layer = nn.Linear(6000, 2)  # Input size should be the sum of token and graph outputs
    def forward(self, dataTokens, dataGraph=None):
        # Token embeddings
        if dataTokens.ndim == 2:
            dataTokens = self.embedding(dataTokens)

        # LSTM processing
        lstm_out, _ = self.lstm(dataTokens)
        x_tokens = lstm_out.mean(dim=1)
        print("Pooled token shape:", x_tokens.shape)  # Should be [batch_size, 12000]

        # Reduce token feature size
        x_tokens = F.relu(self.fc_tokens(x_tokens))  # Shape should be [batch_size, 3000]

        # Process graph data
        if dataGraph is not None:
            features = dataGraph.ndata['type']
            edge_types = dataGraph.edata['label']

            outputs = self.ggnn(dataGraph, features, edge_types)
            print(f'GGNN output shape: {outputs.shape}')

            dataGraph.ndata['GGNNOUTPUT'] = outputs
            x_i, h_i = self.process_graph(dataGraph)

            # Pooling for graph data
            # if isinstance(h_i, list):
            #     x_graph = torch.mean(torch.stack(h_i), dim=0)  # Average across graphs
            # else:
            #     x_graph = torch.mean(h_i, dim=0)  # If h_i is already a tensor
            x_graph = dgl.mean_nodes(dataGraph, 'GGNNOUTPUT')  # Pool to [batch_size, 256]

            # Ensure x_graph has the expected dimension
            print(f'x_graph.shape[-1] != 3000, got {x_graph.shape[-1]}')
            x_graph = self.fc_graph(x_graph)  # Use fully connected layer to expand to 3000
            x_graph = F.relu(x_graph)  # Apply activation function
            print(f'Reshaped graph features: {x_graph.shape}')  # Should be [batch_size, 3000]
            
            
            # Ensure graph features have the same batch size as token features
            if x_graph.size(0) < x_tokens.size(0):
                # Pad or repeat the graph features to match the token batch size
                diff = x_tokens.size(0) - x_graph.size(0)
                x_graph = torch.cat([x_graph, x_graph.new_zeros(diff, x_graph.size(1))], dim=0)
                print(f'Padded graph features: {x_graph.shape}')  # Now should match [batch_size, 3000]

            elif x_graph.size(0) > x_tokens.size(0):
                x_graph = x_graph[:x_tokens.size(0), :]  # Truncate graph features if they exceed the token batch size
                print(f'Truncated graph features: {x_graph.shape}')
            # Concatenate graph and token features
            combined_features = torch.cat((x_tokens, x_graph), dim=1)
        else:
            zero_graph = torch.zeros(x_tokens.size(0), 3000, device=x_tokens.device)
            combined_features = torch.cat((x_tokens, zero_graph), dim=1)

        # Final output layer
        output = self.output_layer(combined_features)

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
