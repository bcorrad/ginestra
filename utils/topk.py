import torch 
from collections import defaultdict

def top_k_accuracy(output, target, k=3):
    """
    Multi-label top-k accuracy for one-hot encoded targets.

    Args:
        output (Tensor): Logits or probabilities of shape (batch_size, num_classes)
        target (Tensor): One-hot encoded labels of shape (batch_size, num_classes)
        k (int): Top-k value (e.g. 1, 3, 5)

    Returns:
        float: Top-k accuracy across the batch
    """
    with torch.no_grad():
        # Get top-k predicted class indices
        topk_preds = output.topk(k, dim=1).indices  # [batch_size, k]

        # Get indices of true labels (where target == 1)
        true_label_indices = target.nonzero(as_tuple=False)  # shape: (n_true_labels, 2)

        # Map sample index to its set of true class indices
        true_dict = defaultdict(set)
        for i, j in true_label_indices:
            true_dict[i.item()].add(j.item())

        # Check if top-k preds intersect with true labels
        correct = torch.zeros(output.size(0), dtype=torch.bool, device=output.device)
        for i in range(output.size(0)):
            pred_set = set(topk_preds[i].tolist())
            if len(pred_set & true_dict[i]) > 0:
                correct[i] = True

        return correct.float().mean().item()


def top_k_coverage(output, target, k=3):
    """
    Calcola la copertura media delle etichette vere nelle top-k predette.
    """
    with torch.no_grad():
        topk_preds = output.topk(k, dim=1).indices
        true_label_indices = target.nonzero(as_tuple=False)
        true_dict = defaultdict(set)
        for i, j in true_label_indices:
            true_dict[i.item()].add(j.item())

        coverage_scores = torch.zeros(output.size(0), device=output.device)
        for i in range(output.size(0)):
            pred_set = set(topk_preds[i].tolist())
            true_set = true_dict[i]
            if len(true_set) > 0:
                coverage_scores[i] = len(pred_set & true_set) / len(true_set)

        return coverage_scores.mean().item()
