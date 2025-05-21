import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, num_node_features, dim_h, num_classes, dim_h_last=512, **kwargs):   #, num_heads=4
        super(GIN, self).__init__()
        
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.dim_h = dim_h
        self.dim_h_last = dim_h_last
        
        # === INPUT LAYER === 
        self.conv1 = GINConv(Sequential(Linear(self.num_node_features, self.dim_h), 
                                        ReLU(),
                                        Linear(self.dim_h, self.dim_h),
                                        ))
        self.bn1 = BatchNorm1d(self.dim_h)
        # === HIDDEN LAYER #1 ===
        self.conv2 = GINConv(Sequential(Linear(self.dim_h, self.dim_h), 
                                        ReLU(),
                                        Linear(self.dim_h, self.dim_h), 
                                        ))
        self.bn2 = BatchNorm1d(self.dim_h)
        # === HIDDEN LAYER #2 ===
        self.conv3 = GINConv(Sequential(Linear(self.dim_h, self.dim_h_last), 
                                        ReLU(),
                                        Linear(self.dim_h_last, self.dim_h_last),
                                        ))
        self.bn3 = BatchNorm1d(self.dim_h_last)

        # Dropout
        if "drop_rate" in kwargs and kwargs["drop_rate"] is not None:
            self.dropout = kwargs["drop_rate"]
            print(f"[DROPOUT SET] Dropout: {self.dropout}")
        else:
            raise ValueError("Dropout rate not specified in kwargs")
        
        self.readout_dim = self.dim_h + self.dim_h + self.dim_h_last + 6144  # h1 + h2 + h3
        
        # === OUTPUT CLASSIFIER ===
        self.out_lin1 = torch.nn.Linear(self.readout_dim, self.readout_dim//2)
        self.out_bn1 = torch.nn.BatchNorm1d(self.out_lin1.out_features)
        self.out_lin3 = torch.nn.Linear(self.out_bn1.num_features, self.num_classes)


    def forward(self, x, edge_index, batch, **kwargs):

        # Node embeddings 
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

        # === Graph-level readout ===
        # The authors make two important points about graph-level readout:
        # To consider all structural information, it is necessary to keep embeddings from previous layers;
        # The sum operator is surprisingly more expressive than the mean and the max.
        h1_pool = global_add_pool(h1, batch)
        h2_pool = global_add_pool(h2, batch)
        h3_pool = global_add_pool(h3, batch)
        h = torch.cat([h1_pool, h2_pool, h3_pool], dim=1)
        # === Add fingerprint ===
        if "fingerprint" in kwargs and kwargs["fingerprint"] is not None:
            h = torch.cat([h, kwargs["fingerprint"]], dim=1)
            
        # Classifier
        h = self.out_lin1(h)
        h = self.out_bn1(h)
        h = F.relu(h)
        # h = self.out_lin2(h)
        # h = self.out_bn2(h)
        # h = F.relu(h)
        h = self.out_lin3(h)
        
        return h



