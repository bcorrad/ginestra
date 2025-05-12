class GIN(torch.nn.Module):
    def __init__(self, num_node_features, dim_h, num_classes, **kwargs):
        super(GIN, self).__init__()

        self.fingerprint_processor = None
        if "fingerprint_length" in kwargs and kwargs["fingerprint_length"] is not None:
            self.fingerprint_processor = torch.nn.Sequential(
                Linear(kwargs["fingerprint_length"], dim_h),
                ReLU(),
                Linear(dim_h, dim_h)
            )

        self.dropout = kwargs.get("drop_rate", 0.3)  # aumenta un po’ per contrastare overfitting
        print(f"[DROPOUT SET] Dropout: {self.dropout}")

        self.conv1 = GINConv(self._gin_mlp(num_node_features, dim_h))
        self.conv2 = GINConv(self._gin_mlp(dim_h, dim_h))
        self.conv3 = GINConv(self._gin_mlp(dim_h, 512))

        self.norm1 = BatchNorm1d(dim_h)
        self.norm2 = BatchNorm1d(dim_h)
        self.norm3 = BatchNorm1d(512)

        readout_dim = dim_h + dim_h + 512
        if self.fingerprint_processor is not None:
            readout_dim += dim_h

        self.lin1 = Linear(readout_dim, 1024)
        self.lin2 = Linear(1024, num_classes)

    def _gin_mlp(self, input_dim, output_dim):
        return Sequential(
            Linear(input_dim, output_dim),
            BatchNorm1d(output_dim),
            ReLU(),
            Dropout(p=self.dropout),
            Linear(output_dim, output_dim),
            ReLU(),
            Dropout(p=self.dropout)
        )

    def forward(self, x, edge_index, batch, **kwargs):
        if "fingerprint" in kwargs:
            fingerprint = kwargs["fingerprint"]
            fingerprint_emb = self.fingerprint_processor(torch.Tensor(fingerprint))
        else:
            fingerprint = None

# Node embeddings 
        h1 = self.conv1(x, edge_index)
        # Dropout 
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.conv2(h1, edge_index)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h3 = self.conv3(h2, edge_index)

# === Graph-level readout ===
        # The authors make two important points about graph-level readout:
        # To consider all structural information, it is necessary to keep embeddings from previous layers;
        # The sum operator is surprisingly more expressive than the mean and the max.
        h1_pool = global_add_pool(h1, batch)
        h2_pool = global_add_pool(h2, batch)
        h3_pool = global_add_pool(h3, batch)

        #Concatenate the embeddings and the fingerprint if not None
        if fingerprint is not None:
            h = torch.cat([h1_pool, h2_pool, h3_pool, fingerprint_emb], dim=1)
        else:
            h = torch.cat([h1_pool, h2_pool, h3_pool], dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = self.lin2(h)

# Config da provare
        # h1 = self.conv1(x, edge_index)
        # h1 = self.norm1(h1)
        # h1_pool = global_mean_pool(h1, batch)

        # h2 = self.conv2(h1, edge_index)
        # h2 = self.norm2(h2)
        # h2_pool = global_mean_pool(h2, batch)

        # h3 = self.conv3(h2, edge_index)
        # h3 = self.norm3(h3)
        # h3_pool = global_mean_pool(h3, batch)

        # h = torch.cat([h1_pool, h2_pool, h3_pool], dim=1)
        # if fingerprint_emb is not None:
        #     h = torch.cat([h, fingerprint_emb], dim=1)

        # h = F.relu(self.lin1(h))
        # h = F.dropout(h, p=self.dropout, training=self.training)
        # h = self.lin2(h)

        return h