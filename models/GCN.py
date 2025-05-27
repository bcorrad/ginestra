import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, ModuleList

class GCN(torch.nn.Module):
    """Parametrized GIN"""
    def __init__(self, num_node_features, dim_h, num_classes, num_layers=5, dim_h_last=512, classifier_hidden_dims=[1024], **kwargs):
        super(GCN, self).__init__()

        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.dim_h = dim_h
        self.dim_h_last = dim_h_last
        self.num_layers = num_layers
        self.readout_dim =  0    # 6144 if kwargs.get("fingerprint") is True else 0
        self.dropout = kwargs.get("drop_rate", ValueError("Dropout rate not specified in kwargs"))

        self.convs = ModuleList()
        self.bns = ModuleList()

        for i in range(num_layers):
            in_dim = self.num_node_features if i == 0 else (dim_h if i < num_layers - 1 else dim_h)
            out_dim = dim_h if i < num_layers - 1 else dim_h_last

            self.convs.append(GCNConv(in_dim, out_dim, cached=False))
            self.bns.append(BatchNorm1d(out_dim))

        self.readout_dim += dim_h * (num_layers - 1) + dim_h_last

        # Classifier
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [self.readout_dim // 2]

        self.classifier = self.build_classifier(self.readout_dim, classifier_hidden_dims, self.num_classes)

    def build_classifier(self, input_dim, hidden_dims, output_dim):
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(Linear(in_dim, hidden_dim))
            layers.append(BatchNorm1d(hidden_dim))
            layers.append(ReLU())
            in_dim = hidden_dim
        layers.append(Linear(in_dim, output_dim))
        return Sequential(*layers)

    def forward(self, x, edge_index, batch, **kwargs):
        xs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            xs.append(global_mean_pool(x, batch))

        #x = torch.cat(xs + [kwargs["fingerprint"]], dim=1) if "fingerprint" in kwargs and kwargs["fingerprint"] is not None else torch.cat(xs, dim=1)
        x = torch.cat(xs, dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x
