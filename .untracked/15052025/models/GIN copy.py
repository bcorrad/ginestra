import torch
import torch.nn.functional as F
import numpy as np
from config import PATHWAYS, TARGET_MODE
import os
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from utils.topk import top_k_accuracy

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, num_node_features, dim_h, num_classes, **kwargs):   #, num_heads=4
        super(GIN, self).__init__()
            
        self.conv1 = GINConv(
            Sequential(Linear(num_node_features, dim_h),
                       #BatchNorm1d(dim_h), 
                       ReLU(),
                       Linear(dim_h, dim_h), 
                       #ReLU()
                       ))
  
        self.bn1 = BatchNorm1d(dim_h)

        self.conv2 = GINConv(Sequential(Linear(dim_h, dim_h), 
                       #BatchNorm1d(dim_h), 
                       ReLU(),
                       Linear(dim_h, dim_h), 
                       #ReLU(
                       ))
        self.bn2 = BatchNorm1d(dim_h)

        self.conv3 = GINConv(Sequential(Linear(dim_h, 512), 
                                        #BatchNorm1d(512), 
                                        ReLU(),
                                        # Linear(512, 512), 
                                        #ReLU()
                                        ))
        self.bn3 = BatchNorm1d(512)
        # Dropout
        if "drop_rate" in kwargs and kwargs["drop_rate"] is not None:
            self.dropout = kwargs["drop_rate"]
        else:
            self.dropout = 0.1

        print(f"[DROPOUT SET] Dropout: {self.dropout}")
        readout_dim = dim_h + dim_h + 512  # h1 + h2 + h3
        self.lin1 = torch.nn.Linear(readout_dim, 1024)
        self.lin2 = torch.nn.Linear(1024, num_classes)


    def forward(self, x, edge_index, batch, **kwargs):
        
        if "fingerprint" in kwargs:
            fingerprint = kwargs["fingerprint"]
            fingerprint_emb = self.fingerprint_processor(torch.Tensor(fingerprint))
        else:
            fingerprint = None

        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        # Dropout 
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

        # Concatenate the embeddings and the fingerprint if not None
        if fingerprint is not None:
            h = torch.cat([h1_pool, h2_pool, h3_pool, fingerprint_emb], dim=1)
        else:
            h = torch.cat([h1_pool, h2_pool, h3_pool], dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = self.lin2(h)
        
        return h


def train_epoch(model, dataloader, optimizer, criterion, device):
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
    all_preds = []
    all_targets = []

    for b, batch in enumerate(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, edge_index=batch.edge_index, batch=batch.batch)
        targets = batch.y
        loss = criterion(out, targets)
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


def evaluate(model, dataloader, device):
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
    all_preds = []
    all_targets = []
    top_k_accuracy_dict = {}

    with torch.no_grad():
        for b, batch in enumerate(dataloader):
            batch = batch.to(device)
            out = model(batch.x, edge_index=batch.edge_index, batch=batch.batch)
            targets = batch.y
            loss = F.cross_entropy(out, targets)
            total_loss += loss.item()
            preds_argmax = torch.argmax(out, dim=1)
            targets_argmax = torch.argmax(targets, dim=1)
            all_preds.extend(preds_argmax.cpu().numpy())
            all_targets.extend(targets_argmax.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return avg_loss, precision, recall, f1, top_k_accuracy_dict


# def evaluate(model, dataloader, device, criterion, epoch_n, return_model=False, save_all_models=False, experiment_folder=None):
#     """
#     Evaluates the model on the given dataloader.
    
#     Args:
#     model: the trained model
#     dataloader: the validation or test dataloader
#     device: the device to use (cpu or cuda)
#     target_type: specifies which target to use ("pathway", "superclass", "class")
    
#     Returns:
#     precision, recall, f1-score
#     """
#     model.eval()
#     total_loss = 0.0
#     all_preds = []
#     all_targets = []
#     top_k_accuracy_dict = {}
#     all_outs = []
#     EXPERIMENT_FOLDER = experiment_folder if experiment_folder is not None else os.path.join(os.getcwd(), "experiments")
    
#     with torch.no_grad():
#         for batch in dataloader:
#             batch = batch.to(device)
#             out = model(batch.x, edge_index=batch.edge_index, batch=batch.batch) # shape [batch_size, num_classes]
#             targets = batch.y # shape [batch_size, num_classes]
#             loss = criterion(out, targets)
#             total_loss += loss.item()
#             preds_argmax = torch.argmax(out, dim=1) # shape [batch_size]
#             targets_argmax = torch.argmax(targets, dim=1) # shape [batch_size]
#             all_preds.extend(preds_argmax.cpu().numpy())
#             all_targets.extend(targets_argmax.cpu().numpy())
#             all_outs.extend(out.cpu().numpy())
    
#     avg_loss = total_loss / len(dataloader)
#     precision = precision_score(all_targets, all_preds, average='macro')
#     recall = recall_score(all_targets, all_preds, average='macro')
#     f1 = f1_score(all_targets, all_preds, average='macro')
#     top_k_accuracy_dict["top_1"] = top_k_accuracy(torch.tensor(np.array(all_outs)), torch.tensor(np.array(all_targets)), k=1)
#     top_k_accuracy_dict["top_3"] = top_k_accuracy(torch.tensor(np.array(all_outs)), torch.tensor(np.array(all_targets)), k=3)
#     top_k_accuracy_dict["top_5"] = top_k_accuracy(torch.tensor(np.array(all_outs)), torch.tensor(np.array(all_targets)), k=5)
    
#     try:
#         conf_matrix = confusion_matrix(np.argmax(all_targets, axis=1), np.argmax(all_preds, axis=1))
#         # print("Validation Confusion Matrix")
#         # print(conf_matrix)
#     except:
#         conf_matrix = None
        
#     # Class-wise metrics (precision, recall, f1 score)
#     print(classification_report(all_targets, all_preds, target_names=PATHWAYS.keys()))
    
#     # Save model if needed
#     if save_all_models:
#         try:
#             torch.save(model.state_dict(), os.path.join(EXPERIMENT_FOLDER, "pt", f"eval_{epoch_n}_{model.__class__.__name__}.pt"))
#         except:
#             print("Error saving model")
    
#     if not return_model:
#         return avg_loss, precision, recall, f1, conf_matrix, top_k_accuracy_dict
#     else:
#         return avg_loss, precision, recall, f1, conf_matrix, model, top_k_accuracy_dict