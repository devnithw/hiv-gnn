import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
torch.manual_seed(42)

        
class GCNConvNetwork(torch.nn.Module):
    def __init__(self, feature_size):
        """
        Graph Neural Network model made with GCN Convolution Layers
        Consists of three layers of GCNConv layers, followed by two Linear layers.
        Output is two classes.
        Parameters - feature_size

        """
        super(GCNConvNetwork, self).__init__()
        
        # Hardcoded parameters
        num_classes = 2
        hidden_channels = 512

        # Layers
        self.conv1 = GCNConv(feature_size, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.linear1 = Linear(hidden_channels, 64)
        self.linear2 = Linear(64, num_classes)


    def forward(self, x, edge_attr, edge_index, batch_index):
        #GCN block (Obtain node embeddings)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # Readout layer
        x = gap(x, batch_index)
        
        # Linear block
        x = self.linear1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)

        return x

