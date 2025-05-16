import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATConv, global_add_pool

class GAT(torch.nn.Module):
    """
    Graph Attention Network with edge features.

    """
    
    def __init__(self, num_node_features, dim_h, num_classes, dim_h_last=512, edge_dim=None, n_heads=4, **kwargs):
        super().__init__()
        
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.dim_h = dim_h
        self.dim_h_last = dim_h_last
        self.edge_dim = edge_dim
        self.n_heads = n_heads
        
        self.conv1 = GATConv(self.num_node_features, dim_h, heads=n_heads, concat=False, edge_dim=self.edge_dim)   # Output (batch_size, dim_h * heads)
        self.bn1 = BatchNorm1d(dim_h)
        self.conv2 = GATConv(self.dim_h, self.dim_h, heads=self.n_heads, concat=False, edge_dim=self.edge_dim)   # Output (batch_size, dim_h * heads)
        self.bn2 = BatchNorm1d(dim_h)
        self.conv3 = GATConv(self.dim_h, self.dim_h_last, heads=self.n_heads, concat=False, edge_dim=self.edge_dim)   # Output (batch_size, dim_h * heads)
        self.bn3 = BatchNorm1d(self.dim_h_last)

        # Dropout
        if "drop_rate" in kwargs and kwargs["drop_rate"] is not None:
            self.dropout = kwargs["drop_rate"]
        else:
            raise ValueError("Dropout rate not specified in kwargs")

        print(f"[DROPOUT SET] Dropout: {self.dropout}")
        # Final classifier
        self.readout_dim = self.dim_h + self.dim_h + self.dim_h_last

        self.fc1 = torch.nn.Linear(self.readout_dim, 1024)
        self.fc2 = torch.nn.Linear(1024, self.num_classes)

    def forward(self, x, edge_index, batch, **kwargs):

        # Strati GINEConv
        # GAT + BatchNorm + ReLU + Dropout
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        h3 = self.conv3(h2, edge_index)
        h3 = self.bn3(h3)
        h3 = F.relu(h3)

        # Global pooling
        h1_pool = global_add_pool(h1, batch)
        h2_pool = global_add_pool(h2, batch)
        h3_pool = global_add_pool(h3, batch)

        h = torch.cat([h1_pool, h2_pool, h3_pool], dim=1)

        # Classificatore
        h = self.fc1(h).relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fc2(h)

        return h   
    