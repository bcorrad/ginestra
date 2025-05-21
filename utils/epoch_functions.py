import torch, numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from utils.topk import top_k_accuracy, top_k_coverage
from config import USE_FINGERPRINT


def training_epoch(model, dataloader, optimizer, criterion, device):
    """
    Training loop for the model.
    Args:
    model: the model to train
    dataloader: the training dataloader
    optimizer: the optimizer to use
    criterion: the loss function
    device: the device to use (cpu or cuda)
    cumulative_loss: if True, the loss will be the sum of the CrossEntropyLoss and the cosine similarity loss

    Returns:
    avg_loss: the average loss over the training set
    loss: the loss of the last batch

    """
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for b, batch in enumerate(dataloader):
        if "MLP" in model.__class__.__name__:
            # Fingerprint at index 1 (shape [batch, 1, feature_dim]); target at index 2
            x, y = batch[1].to(device).float(), batch[2].to(device).float()
            x = x.squeeze(1) if x.dim() == 3 else x  # [batch, feature_dim]
            optimizer.zero_grad()
            out = model(x)
        elif "GIN" in model.__class__.__name__ or "GAT" in model.__class__.__name__:
            x, y, edge_index, batch_ = batch.x.to(device).float(), batch.y.to(device).float(), batch.edge_index.to(device), batch.batch.to(device)
            optimizer.zero_grad()
            out = model(x, edge_index=edge_index, batch=batch_, fingerprint=batch.fingerprint_tensor.to(device) if USE_FINGERPRINT else None) # [batch, num_classes]
        loss = criterion(out, y.argmax(dim=1)) if not isinstance(criterion, torch.nn.BCEWithLogitsLoss) else criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(out.detach().cpu().numpy())
        all_targets.extend(y.detach().cpu().numpy())
        
    avg_loss = total_loss / len(dataloader)
    precision = precision_score(all_targets, (F.sigmoid(torch.Tensor(np.array(all_preds)))>0.5), average='macro', zero_division=0)
    recall = recall_score(all_targets, (F.sigmoid(torch.Tensor(np.array(all_preds)))>0.5), average='macro', zero_division=0)
    f1 = f1_score(all_targets, (F.sigmoid(torch.Tensor(np.array(all_preds)))>0.5), average='macro', zero_division=0)

    return avg_loss, precision, recall, f1


def evaluation_epoch(model, dataloader, criterion, device):
    """
    Evaluation loop for the model.
    Args:
    model: the model to evaluate
    dataloader: the evaluation dataloader
    device: the device to use (cpu or cuda)
    verbose: if True, print the classification report

    Returns:
    avg_loss: the average loss over the evaluation set
    loss: the loss of the last batch
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    topk = {}

    with torch.no_grad():
        for b, batch in enumerate(dataloader):
            if "MLP" in model.__class__.__name__:
                # Fingerprint at index 1; target at index 2
                x, y = batch[1].to(device).float(), batch[2].to(device).float()
                x = x.squeeze(1) if x.dim() == 3 else x  # [batch, feature_dim]
                out = model(x)
            elif "GIN" in model.__class__.__name__ or "GAT" in model.__class__.__name__:
                x, y, edge_index, batch_ = batch.x.to(device).float(), batch.y.to(device).float(), batch.edge_index.to(device), batch.batch.to(device)
                out = model(x, edge_index=edge_index, batch=batch_, fingerprint=batch.fingerprint_tensor.to(device) if USE_FINGERPRINT else None) # [batch, num_classes]
            loss = criterion(out, y.argmax(dim=1)) if not isinstance(criterion, torch.nn.BCEWithLogitsLoss) else criterion(out, y)
            total_loss += loss.item()
            all_preds.extend(out.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        precision = precision_score(all_targets, (F.sigmoid(torch.Tensor(np.array(all_preds))) > 0.5), average='macro', zero_division=0)
        recall = recall_score(all_targets, (F.sigmoid(torch.Tensor(np.array(all_preds))) > 0.5), average='macro', zero_division=0)
        f1 = f1_score(all_targets, (F.sigmoid(torch.Tensor(np.array(all_preds))) > 0.5), average='macro', zero_division=0)
        topk['1'] = top_k_accuracy(out, y, k=1)
        topk['3'] = top_k_accuracy(out, y, k=3)
        topk['5'] = top_k_accuracy(out, y, k=5)
        topk['1_coverage'] = top_k_coverage(out, y, k=1)
        topk['3_coverage'] = top_k_coverage(out, y, k=3)
        topk['5_coverage'] = top_k_coverage(out, y, k=5)

    return avg_loss, precision, recall, f1, topk
