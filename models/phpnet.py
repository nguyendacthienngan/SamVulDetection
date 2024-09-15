import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torch_geometric.nn import GCNConv, SAGEConv, DNAConv, ARMAConv, ChebConv, GINConv, GatedGraphConv, SplineConv, TopKPooling, GATConv, EdgePooling, TAGConv,DynamicEdgeConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, global_sort_pool


class PhpNetGraphTokensCombine(nn.Module):
    def __init__(self):
        super(PhpNetGraphTokensCombine, self).__init__()
        self.embed1 = nn.Embedding(num_embeddings=5000,
                                  embedding_dim=100)
        self.conv1 = GCNConv(2000,2000)
        self.pool1 = EdgePooling(2000)
        self.conv2 = GCNConv(2000, 4000)
        self.pool2 = EdgePooling(4000)
        self.conv3 = GCNConv(4000, 4000)
        self.pool3 = EdgePooling(4000)
        self.conv4 = GCNConv(2000, 2000)
        self.pool4 = EdgePooling(8000)

        #
        self.embed = nn.Embedding(num_embeddings=5000,
                                  embedding_dim=100)
        self.lstm1 = nn.GRU(input_size=100,
                            hidden_size=200,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=True)



        self.lin1 = nn.Linear(5200, 1000)
        self.lin11 = nn.Linear(1000, 500)
        self.lin2 = nn.Linear(500, 4)
        self.lin3 = nn.Linear(200, 100)
        self.lin4 = nn.Linear(100,4)

    def forward(self, dataGraph, dataTokens):
        x, edge_index = dataGraph.x.long(), dataGraph.edge_index
        batch= dataGraph.batch
        pre_x_len = len(x)
        x = self.embed1(x)
        x = x.reshape(pre_x_len,-1)
        x = self.conv1(x, edge_index)
        x, edge_index, batch,_ = self.pool1(x,edge_index,batch=dataGraph.batch)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training, p=0.3)

        # # x = F.dropout(x, training=self.training,p=0.5)
        #
        x = self.conv2(x, edge_index)
        x, edge_index, batch, _= self.pool2(x,edge_index,batch=batch)
        x = F.relu(x)

        # # x = F.leaky_relu(x,0.2)
        # # x = F.dropout(x, training=self.training, p=0.5)
        #
        x = self.conv3(x, edge_index)
        x, edge_index, batch, _= self.pool3(x, edge_index, batch=batch)
        x = F.relu(x)


        x = global_max_pool(x, batch)


        x1 = self.embed(dataTokens)
        output1, (hidden1) = self.lstm1(x1)
        x1 = torch.cat((hidden1[0, :, :],hidden1[1, :, :],hidden1[2, :, :],hidden1[-3, :, :], hidden1[-2, :, :], hidden1[-1, :, :]), dim=1)
        # x1 = F.relu(self.fc1(x1))
        # x1 = F.relu(self.fc2(x1))
        x = torch.cat([x,x1], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin11(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin2(x))
        return x

