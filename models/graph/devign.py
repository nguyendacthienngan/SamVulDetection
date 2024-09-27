import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GatedGraphConv as DGLGatedGraphConv
from torch_geometric.nn.conv import GatedGraphConv as PyGGatedGraphConv  # Import from torch_geometric
import dgl


class ModelParser:
    vector_size: int = 100  # Graph node vector dimension
    hidden_size: int = 256  # GNN hidden layer vector dimension
    layer_num: int = 5  # Number of GNN layers
    num_classes: int = 2

model_args = ModelParser()


class DevignModel(nn.Module):
    def __init__(self, num_layers=1, max_edge_types=6, num_steps=6):
        super(DevignModel, self).__init__()
        input_dim = max_edge_types + model_args.vector_size
        output_dim = model_args.hidden_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps

        # Initialize both versions of GatedGraphConv
        self.ggnn = DGLGatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps, n_etypes=max_edge_types)
        self.ggnn_exp = PyGGatedGraphConv(out_channels=output_dim, num_layers=num_steps)

        self.conv_l1 = nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = nn.MaxPool1d(3, stride=2)
        self.conv_l2 = nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim  # 106 + 256 = 362
        self.conv_l1_for_concat = nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

        # Attribute to track mode
        self.is_explainer_mode = False

    def set_explainer_mode(self, mode: bool):
        """
        Set the mode for explainability.
        Args:
            mode (bool): True for explainer mode, False for training mode.
        """
        self.is_explainer_mode = mode

    def forward(self, node_features, edge_index, batch, edge_types=None):
        """
        Refactored forward function to accept node_features and edge_index directly.
        """
        # Convert edge_index to a DGL graph
        src, dst = edge_index  # assuming edge_index is a tuple of (source, destination)
        graph = dgl.graph((src, dst), num_nodes=node_features.size(0))

        if self.is_explainer_mode:
            # Use PyTorch Geometric GatedGraphConv for explainer mode
            outputs = self.ggnn_exp(node_features, edge_index)

            # Adjust the outputs to match the shape (batch_size, num_classes)
            aligned_outputs = outputs.view(2, 2)
            return aligned_outputs
            # x_i, h_i = self.unbatch_features_with_batch(node_features, outputs, batch)
        else:
            # Use DGL GatedGraphConv for regular mode
            outputs = self.ggnn(graph, node_features, edge_types)
            batch.ndata['GGNNOUTPUT'] = outputs
            x_i, h_i = self.unbatch_features(batch)

        x_i = torch.stack(x_i)
        h_i = torch.stack(h_i)
        c_i = torch.cat((h_i, x_i), dim=-1)

        Y_1 = self.maxpool1(F.relu(self.conv_l1(h_i.transpose(1, 2))))
        Y_2 = self.maxpool2(F.relu(self.conv_l2(Y_1))).transpose(1, 2)

        Z_1 = self.maxpool1_for_concat(F.relu(self.conv_l1_for_concat(c_i.transpose(1, 2))))
        Z_2 = self.maxpool2_for_concat(F.relu(self.conv_l2_for_concat(Z_1))).transpose(1, 2)

        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        temp = torch.cat((Y_2.sum(1), Z_2.sum(1)), 1)
        result = self.sigmoid(avg).squeeze(dim=-1)

        return result, avg, temp

    def unbatch_features(self, g):
        x_i = []
        h_i = []
        max_len = -1

        for g_i in dgl.unbatch(g):
            x_i.append(g_i.ndata['type'])
            h_i.append(g_i.ndata['GGNNOUTPUT'])
            max_len = max(g_i.number_of_nodes(), max_len)

        for i, (v, k) in enumerate(zip(x_i, h_i)):
            if v.dim() == 1:
                v = v.unsqueeze(1)
            if k.dim() == 1:
                k = k.unsqueeze(1)

            if v.size(1) != self.input_dim:
                v = F.pad(v, (0, self.input_dim - v.size(1)), value=0)
            if k.size(1) != self.output_dim:
                k = F.pad(k, (0, self.output_dim - k.size(1)), value=0)

            x_i[i] = torch.cat((v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad,
                                               device=v.device)), dim=0)
            h_i[i] = torch.cat((k, torch.zeros(size=(max_len - k.size(0), *(k.shape[1:])), requires_grad=k.requires_grad,
                                               device=k.device)), dim=0)

        return x_i, h_i

    def unbatch_features_with_batch(self, node_features, ggnn_outputs, batch=None):
        """
        Refactored unbatch_features to handle node_features and GGNN outputs directly.
        Args:
            node_features: The original node features.
            ggnn_outputs: The outputs from the GGNN.
            batch: Optional; a tensor representing the batch indices, used in explainer mode.
        """
        x_i = []
        h_i = []
        max_len = -1

        for i in range(len(node_features)):
            x_i.append(node_features[i])
            h_i.append(ggnn_outputs[i])
            max_len = max(node_features[i].size(0), max_len)

        for i, (v, k) in enumerate(zip(x_i, h_i)):
            if v.size(1) != self.input_dim:
                v = F.pad(v, (0, self.input_dim - v.size(1)), value=0)
            if k.size(1) != self.output_dim:
                k = F.pad(k, (0, self.output_dim - k.size(1)), value=0)

            if batch is not None:
                x_i[i] = torch.cat((v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad,
                                                   device=v.device)), dim=0)
                h_i[i] = torch.cat((k, torch.zeros(size=(max_len - k.size(0), *(k.shape[1:])), requires_grad=k.requires_grad,
                                                   device=k.device)), dim=0)
            else:
                x_i[i] = torch.cat((v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad,
                                                   device=v.device)), dim=0)
                h_i[i] = torch.cat((k, torch.zeros(size=(max_len - k.size(0), *(k.shape[1:])), requires_grad=k.requires_grad,
                                                   device=k.device)), dim=0)

        return x_i, h_i
