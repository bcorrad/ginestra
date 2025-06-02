import torch

def normalize_adjacency(adj):
    """
    Normalizes the adjacency matrix using the symmetric normalization:
    Ĥ = D^{-1/2} (A + I) D^{-1/2}
    """
    # Add self-connections
    adj = adj + torch.eye(adj.size(0), device=adj.device)
    
    # Degree matrix
    deg = adj.sum(dim=1)
    
    # D^{-1/2}
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    
    # Normalized adjacency
    return D_inv_sqrt @ adj @ D_inv_sqrt

def is_normalized(adj, tol=1e-5):
    """
    Checks if the adjacency matrix is approximately symmetric and values are in [0, 1].
    """
    symmetric = torch.allclose(adj, adj.T, atol=tol)
    in_range = adj.min() >= 0 and adj.max() <= 1
    return symmetric and in_range

def oversmoothing_test(adjacency_matrix, h1, layer_name):
    """
    Test for oversmoothing in a graph neural network.
    Args:
        adjacency_matrix (torch.Tensor): The adjacency matrix of the graph.
        h1 (torch.Tensor): The output of the first layer of the GNN.
        layer_name (str): The name of the layer being tested.
    """
    # Check and normalize adjacency matrix if needed
    if not is_normalized(adjacency_matrix):
        print("Adjacency matrix not normalized. Normalizing now...")
        adjacency_matrix = normalize_adjacency(adjacency_matrix)
    else:
        print("Adjacency matrix is already normalized.")

    # Oversmoothing test
    oversmoothing = torch.matmul(adjacency_matrix, h1)
    similarity = torch.nn.functional.cosine_similarity(h1, oversmoothing, dim=-1)
    print(f"Similarity between {layer_name} and oversmoothing: {similarity.mean().item()}")
