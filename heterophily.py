import torch
import numpy as np
import math
from collections import defaultdict
from torch_geometric.data import Data

import numpy as np
from collections import defaultdict

import torch
import numpy as np
import math
from collections import defaultdict
from torch_geometric.data import Data

def compute_graph_homophily_metrics(data_list):
    all_nodes = []
    all_labels = []
    all_edges = []

    for data in data_list:
        edge_index = data.edge_index
        y = data.y.squeeze()
        num_nodes = data.x.size(0)

        if y.dim() == 0:
            y = y.unsqueeze(0)

        labels = y.cpu().numpy()
        all_nodes.extend(range(len(all_labels), len(all_labels) + num_nodes))
        all_labels.extend(labels.tolist())

        offset = len(all_labels) - num_nodes
        edge_list = edge_index.cpu().numpy().T.tolist()
        edge_list_offset = [(u + offset, v + offset) for u, v in edge_list]
        all_edges.extend(edge_list_offset)

    all_labels = np.array(all_labels)
    total_nodes = len(all_labels)
    total_edges = len(all_edges)

    # === Node Homophily ===
    neighbor_dict = defaultdict(list)
    for u, v in all_edges:
        neighbor_dict[u].append(v)

    node_homophilies = []
    for node in range(total_nodes):
        neighbors = neighbor_dict[node]
        if len(neighbors) == 0:
            continue
        same_class_neighbors = sum(all_labels[node] == all_labels[n] for n in neighbors)
        node_homophilies.append(same_class_neighbors / len(neighbors))
    H_node = np.mean(node_homophilies)

    # === Edge Homophily ===
    same_class_edges = sum(1 for u, v in all_edges if all_labels[u] == all_labels[v])
    H_edge = same_class_edges / total_edges

    # === Class Homophily ===
    unique_classes = np.unique(all_labels)
    C = len(unique_classes)
    class_homophilies = []
    for c in unique_classes:
        class_nodes = np.where(all_labels == c)[0]
        nc = len(class_nodes)
        if nc == 0:
            continue
        numerator = sum(
            sum(all_labels[n] == c for n in neighbor_dict[v])
            for v in class_nodes
            if len(neighbor_dict[v]) > 0
        )
        denominator = sum(len(neighbor_dict[v]) for v in class_nodes if len(neighbor_dict[v]) > 0)
        p_c = nc / total_nodes
        if denominator > 0:
            class_homophilies.append((numerator / denominator) - p_c)

    H_class = np.mean(class_homophilies) if class_homophilies else 0

    # === Adjusted Homophily ===
    expected_edge_homophily = 0
    for c in unique_classes:
        p_c = sum(all_labels == c) / total_nodes
        expected_edge_homophily += p_c ** 2
    if expected_edge_homophily < 1:
        H_adj = (H_edge - expected_edge_homophily) / (1 - expected_edge_homophily)
    else:
        H_adj = 0

    # === Label Informativeness (approximation) ===
    label_distribution = np.bincount(all_labels) / total_nodes
    H_yu = -np.sum(label_distribution * np.log2(label_distribution + 1e-10))

    cond_label_counts = defaultdict(lambda: defaultdict(int))
    for u, v in all_edges:
        cond_label_counts[all_labels[v]][all_labels[u]] += 1

    cond_probs = {}
    for yv in cond_label_counts:
        total = sum(cond_label_counts[yv].values())
        cond_probs[yv] = {yu: count / total for yu, count in cond_label_counts[yv].items()}

    H_yu_given_yv = 0
    for yv in cond_probs:
        p_yv = np.mean(all_labels == yv)
        entropy = -sum(p * np.log2(p + 1e-10) for p in cond_probs[yv].values())
        H_yu_given_yv += p_yv * entropy

    LI = (H_yu - H_yu_given_yv) / (H_yu + 1e-10)

    return {
        "H_node": round(H_node, 4),
        "H_edge": round(H_edge, 4),
        "H_class": round(H_class, 4),
        "H_adj": round(H_adj, 4),
        "Label_Informativeness": round(LI, 4)
    }



import pickle

with open('/mnt/beegfs/home/giulio/metabolomic/ginestra/data/data/train_geodataloader_MULTILABEL_BCDEF_pathway.pkl', 'rb') as f:
    traindataloader = pickle.load(f)

gnn_train_dataloader = traindataloader["dataloader"]

# Prendi un solo grafo, ad esempio il primo
data = gnn_train_dataloader.dataset[:10]
print(data)

metrics = compute_graph_homophily_metrics(data)
for k, v in metrics.items():
    print(f"{k}: {v}")