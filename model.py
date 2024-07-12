import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import TransformerConv, GATConv, TopKPooling, BatchNorm
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.conv.x_conv import XConv

class GNN(torch.nn.Module):
    def __init__(self, feature_size):
        super(GNN, self).__init__()
        
        # Hardcoded parameters
        num_classes = 2
        embedding_size = 1024

        # Layers
        self.conv1 = GATConv(feature_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform1 = Linear(embedding_size * 3, embedding_size)
        self.pooling1 = TopKPooling(embedding_size, ratio=0.8)
        self.conv2 = GATConv(feature_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform2 = Linear(embedding_size * 3, embedding_size)
        self.pooling2 = TopKPooling(embedding_size, ratio=0.5)
        self.conv3 = GATConv(feature_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform3= Linear(embedding_size * 3, embedding_size)
        self.pooling3 = TopKPooling(embedding_size, ratio=0.2)

        self.linear1 = Linear(embedding_size * 2, 1024)
        self.linear2 = Linear(1024, num_classes)

    def forward(self, x, edge_attr, edge_index, batch_index):
        # First block
        x = self.conv1(x, edge_index)
        x = self.head_transform1(x)
        x, edge_index, edge_attr, batch_index, _, _ = self.pooling1(x, edge_index, None, batch_index)
        x1 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        # Second block
        x = self.conv2(x, edge_index)
        x = self.head_transform2(x)
        x, edge_index, edge_attr, batch_index, _, _= self.pooling2(x, edge_index, None, batch_index)
        x2 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        # Third block
        x = self.conv3(x, edge_index)
        x = self.head_transform3(x)
        x, edge_index, edge_attr, batch_index, _, _= self.pooling3(x, edge_index, None, batch_index)
        x3 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        # Concatenate pooled features
        x = x1 + x2 + x3

        # Linear block
        x = self.linear1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)

        return x

        


