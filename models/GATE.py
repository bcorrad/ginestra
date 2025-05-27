import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, GELU
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn import Set2Set

class GATE(torch.nn.Module):
    """
    Graph Attention Network.

    """
    
    def __init__(self, num_node_features, dim_h, num_classes, dim_h_last=256, edge_dim=None, n_heads_in=4, n_heads_out=1, **kwargs):
        super().__init__()
        
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.dim_h = dim_h
        self.dim_h_last = dim_h_last
        self.edge_dim = edge_dim
        self.n_heads = n_heads_in
        
        
        self.conv1 = GATv2Conv(self.num_node_features, self.dim_h, heads=self.n_heads, concat=True, edge_dim=self.edge_dim)   # Output (batch_size, dim_h * heads)
        self.bn1 = BatchNorm1d(dim_h * n_heads_in)
        self.lin1 = Linear(num_node_features, dim_h * n_heads_in)

        self.conv2 = GATv2Conv(self.dim_h * n_heads_in, self.dim_h, heads=self.n_heads, concat=False, edge_dim=self.edge_dim)   # Output (batch_size, dim_h * heads)
        self.bn2 = BatchNorm1d(dim_h)
        self.lin2 = Linear(dim_h * n_heads_in, dim_h)  # Per skip

        self.conv3 = GATv2Conv(self.dim_h, self.dim_h, heads=self.n_heads, concat=False, edge_dim=self.edge_dim)
        self.bn3 = BatchNorm1d(self.dim_h)
        self.lin3 = Linear(self.dim_h, self.dim_h)  # per skip connection

        # self.conv4 = GATv2Conv(self.dim_h, self.dim_h, heads=self.n_heads, concat=False, edge_dim=self.edge_dim)
        # self.bn4 = BatchNorm1d(self.dim_h)
        # self.lin4 = Linear(self.dim_h, self.dim_h)

        self.conv5 = GATv2Conv(self.dim_h, self.dim_h_last, heads=n_heads_out, concat=False, edge_dim=self.edge_dim)
        self.bn5 = BatchNorm1d(self.dim_h_last)
        self.lin5 = Linear(dim_h, dim_h_last) 

        # Dropout
        if "drop_rate" in kwargs and kwargs["drop_rate"] is not None:
            self.dropout = kwargs["drop_rate"]
        else:
            raise ValueError("Dropout rate not specified in kwargs")

        print(f"[DROPOUT SET] Dropout: {self.dropout}")
        # Final classifier
        #self.readout_dim = self.dim_h + self.dim_h + self.dim_h + self.dim_h_last
        
        # self.pool = Set2Set(self.dim_h_last, processing_steps=3)

        # self.readout_dim = self.dim_h_last * 2  

        
        self.readout_dim = self.dim_h_last

        self.fc1 = torch.nn.Linear(self.readout_dim, 1024)
        self.fc2 = torch.nn.Linear(1024, self.num_classes)

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):

        # Layer 1 + skip
        h1 = self.conv1(x, edge_index, edge_attr)
        h1 = self.bn1(h1)
        h1 = F.elu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h1 = h1 + self.lin1(x)  # skip connection (adattamento dimensione)

        # Layer 2 + skip
        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = self.bn2(h2)
        h2 = F.elu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h2 = h2 + self.lin2(h1)

        #Layer 3 + skip
        h3 = self.conv3(h2, edge_index, edge_attr)
        h3 = self.bn3(h3)
        h3 = F.elu(h3)
        h3 = F.dropout(h3, p=self.dropout, training=self.training)
        h3 = h3 + self.lin3(h2)

        #Layer 4 + skip
        # h4 = self.conv3(h3, edge_index)
        # h4 = self.bn3(h4)
        # h4 = F.elu(h4)
        # h4 = F.dropout(h4, p=self.dropout, training=self.training)
        # h4 = h4 + self.lin3(h3)


        # Layer  + skip
        h5 = self.conv5(h3, edge_index)
        h5 = self.bn5(h5)
        h5 = F.elu(h5)
        h5 = F.dropout(h5, p=self.dropout, training=self.training)
        h5 = h5 + self.lin5(h3)

        # Pooling
        h = global_mean_pool(h5, batch)

        # Classificatore
        h = self.fc1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fc2(h)

        # Global pooling
        # h1_pool = global_add_pool(h1, batch)
        # h2_pool = global_add_pool(h2, batch)
        # h4_pool = global_add_pool(h4, batch)
        # h5_pool = global_add_pool(h5, batch)

        # h = torch.cat([h1_pool, h2_pool, h4_pool, h5_pool], dim=1)


        # Global Max Pooling SOLO sul layer finale
        #h = global_mean_pool(h5, batch)


        return h   
    