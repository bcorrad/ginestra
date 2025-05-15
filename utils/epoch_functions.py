import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


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
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, edge_index=batch.edge_index, batch=batch.batch) # [batch, num_classes]
        targets = batch.y                                                      
        loss = criterion(out, targets.argmax(dim=1)) if not isinstance(criterion, torch.nn.BCEWithLogitsLoss) else criterion(out, targets.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds_argmax = torch.argmax(out, dim=1)
        targets_argmax = torch.argmax(targets, dim=1)
        all_preds.extend(preds_argmax.cpu().numpy())
        all_targets.extend(targets_argmax.cpu().numpy())
        
    avg_loss = total_loss / len(dataloader)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

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

    with torch.no_grad():
        for b, batch in enumerate(dataloader):
            batch = batch.to(device)
            out = model(batch.x, edge_index=batch.edge_index, batch=batch.batch)
            targets = batch.y
            loss = criterion(out, targets.argmax(dim=1)) if not isinstance(criterion, torch.nn.BCEWithLogitsLoss) else criterion(out, targets.float())
            total_loss += loss.item()
            preds_argmax = torch.argmax(out, dim=1)
            targets_argmax = torch.argmax(targets, dim=1)
            all_preds.extend(preds_argmax.cpu().numpy())
            all_targets.extend(targets_argmax.cpu().numpy())
            if b%100 == 0:
                print(f"Batch {b}/{len(dataloader)}, Loss: {loss.item():.4f}, Precision: {precision_score(targets_argmax.cpu(), preds_argmax.cpu(), average='macro', zero_division=0):.4f}, Recall: {recall_score(targets_argmax.cpu(), preds_argmax.cpu(), average='macro', zero_division=0):.4f}, F1: {f1_score(targets_argmax.cpu(), preds_argmax.cpu(), average='macro', zero_division=0):.4f}")

        avg_loss = total_loss / len(dataloader)
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return avg_loss, precision, recall, f1

