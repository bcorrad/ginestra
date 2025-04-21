import torch
import torch.nn.functional as F
import numpy as np
from config import EXPERIMENT_FOLDER
import os
from torch_geometric.nn import GATConv, global_add_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from sklearn.metrics import classification_report
from utils.topk import top_k_accuracy
from config import PATHWAYS
from config import TARGET_MODE
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
metrics_average_mode = "two_classes" if TARGET_MODE == "two_classes" else "micro"
BCE_THRESHOLD = "tanto te ne andrai"

class GATE(torch.nn.Module):
    """
    Graph Attention Network with edge features.

    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, n_heads=4, **kwargs):
        super().__init__()
        if "fingerprint_length" in kwargs and kwargs["fingerprint_length"] is not None:
            self.fingerprint_processor = torch.nn.Sequential(
                                    torch.nn.Linear(kwargs["fingerprint_length"], hidden_channels),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_channels, hidden_channels))
        else:
            self.fingerprint_processor = None
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=n_heads, concat=False, edge_dim=edge_dim)   # Output (batch_size, hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=n_heads, concat=False, edge_dim=edge_dim)   # Output (batch_size, hidden_channels * heads)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=n_heads, concat=False, edge_dim=edge_dim)   # Output (batch_size, hidden_channels * heads)
        
         # Classificatore finale
        if "fingerprint_length" not in kwargs or kwargs["fingerprint_length"] is None:
            self.fc1 = torch.nn.Linear(3*hidden_channels, 3*hidden_channels)  
            self.fc2 = torch.nn.Linear(3*hidden_channels, out_channels)
        else:
            self.fc1 = torch.nn.Linear(4*hidden_channels, 4*hidden_channels)
            self.fc2 = torch.nn.Linear(4*hidden_channels, out_channels)


    def forward(self, x, edge_index, edge_attr, batch, p=0.2, **kwargs):
        
        if "fingerprint" in kwargs:
            fingerprint = kwargs["fingerprint"]
            fingerprint_emb = self.fingerprint_processor(torch.Tensor(fingerprint))
        else:
            fingerprint = None
            
        # # Primo livello: GAT
        # x = self.gat_conv1(x, edge_index)  # Output (batch_size, hidden_channels * heads)
        # x = F.dropout(x, p=0.5, training=self.training)

        # Strati GINEConv
        h1 = self.conv1(x, edge_index, edge_attr)
        h1 = F.dropout(h1, p=p, training=self.training)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = F.dropout(h2, p=p, training=self.training)
        h3 = self.conv3(h2, edge_index, edge_attr)

        # Global pooling on node features
        h1_pool = global_add_pool(h1, batch)
        h2_pool = global_add_pool(h2, batch)
        h3_pool = global_add_pool(h3, batch)
        
        # Concatenate the embeddings and the fingerprint if not None
        if fingerprint is not None:
            h = torch.cat([h1_pool, h2_pool, h3_pool, fingerprint_emb], dim=1)
        else:
            h = torch.cat([h1_pool, h2_pool, h3_pool], dim=1)

        # Classificatore
        h = self.fc1(h).relu()
        h = self.fc2(h)

        return h    
    

def evaluate(model, dataloader, device, criterion, epoch_n, return_model=False, save_all_models=False):
    """
    Evaluates the model on the given dataloader.
    
    Args:
    model: the trained model
    dataloader: the validation or test dataloader
    device: the device to use (cpu or cuda)
    target_type: specifies which target to use ("pathway", "superclass", "class")
    
    Returns:
    precision, recall, f1-score
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    top_k_accuracy_dict = {}
    all_outs = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            if model.__class__.__name__ == "GIN":
                out = model(batch.x, edge_index=batch.edge_index, batch=batch.batch)
            elif model.__class__.__name__ == "GINWithEdgeFeatures" or model.__class__.__name__ == "NewGINE":
                from config import USE_FINGERPRINT
                if USE_FINGERPRINT:
                    out = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch, fingerprint=batch.fingerprint)
                else:
                    out = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
                #out = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch, fingerprint=batch.fingerprint)
            elif model.__class__.__name__ == "GATE":
                out = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
            # out = model(batch.x, batch.edge_index, batch.batch)  # Forward pass
            # Determine targets
            targets = batch.y

            # Compute loss
            loss = criterion(out, targets)
            total_loss += loss.item()
            
            # Apply threshold for binary classification
            max_idx = torch.argmax(F.softmax(out, dim=1), dim=1, keepdim=True)
            preds = torch.zeros_like(out)
            for row_idx, col_idx in enumerate(max_idx):
                preds[row_idx, col_idx] = 1
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_outs.extend(out.cpu().numpy())
    
    # Calcolo della loss media e dell'accuracy totale sul validation set
    avg_loss = total_loss / len(dataloader)
    precision = precision_score(all_targets, all_preds, average=metrics_average_mode)
    recall = recall_score(all_targets, all_preds, average=metrics_average_mode)
    f1 = f1_score(all_targets, all_preds, average=metrics_average_mode)
    top_k_accuracy_dict["top_1"] = top_k_accuracy(torch.tensor(np.array(all_outs)), torch.tensor(np.array(all_targets)), k=1)
    top_k_accuracy_dict["top_3"] = top_k_accuracy(torch.tensor(np.array(all_outs)), torch.tensor(np.array(all_targets)), k=3)
    top_k_accuracy_dict["top_5"] = top_k_accuracy(torch.tensor(np.array(all_outs)), torch.tensor(np.array(all_targets)), k=5)
    try:
        conf_matrix = confusion_matrix(np.argmax(all_targets, axis=1), np.argmax(all_preds, axis=1))
        # print("Validation Confusion Matrix")
        # print(conf_matrix)
    except:
        conf_matrix = None
        
    # Class-wise metrics (precision, recall, f1 score)
    print(classification_report(all_targets, all_preds, target_names=PATHWAYS.keys()))
    
    # Save model if needed
    if save_all_models:
        try:
            torch.save(model.state_dict(), os.path.join(EXPERIMENT_FOLDER, "pt", f"eval_{epoch_n}_{model.__class__.__name__}.pt"))
        except:
            print("Error saving model")
    
    if not return_model:
        return avg_loss, precision, recall, f1, conf_matrix, top_k_accuracy_dict
    else:
        return avg_loss, precision, recall, f1, conf_matrix, model, top_k_accuracy_dict


def train_epoch(model, dataloader, optimizer, criterion, device, epoch_n, verbose:bool=False, return_model=False, save_all_models=False):
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

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        # Forward pass
        out = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)

        # out = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)     # (batch_dim, features_dim) = (128, 653)
        # Targets
        targets = batch.y

        # Compute loss and optimize
        if TARGET_MODE == "two_classes":
            loss = criterion(out, targets.unsqueeze(-1))
        else:
            loss = criterion(out, targets)    # CrossEntropyLoss (fa la logsoftmax in automatico): input=logits per ciascuna classe, labels=tensore di interi con la classe corretta per ogni campione.
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Apply threshold for binary classification
        if TARGET_MODE == "two_classes":
            preds = (F.sigmoid(out) > BCE_THRESHOLD).to(int).view(-1)
            targets = targets.view(-1)
        elif TARGET_MODE == "ohe" or TARGET_MODE == "binary" or "hot" in TARGET_MODE:
            # preds = torch.argmax(F.softmax(out, dim=1))
            max_idx = torch.argmax(F.softmax(out, dim=1), dim=1, keepdim=True)
            preds = torch.zeros_like(out)
            for row_idx, col_idx in enumerate(max_idx):
                preds[row_idx, col_idx] = 1
        else:
            targets = targets.view(-1)
            preds = out.view(-1)
            
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        if verbose:
            print(f"out = {preds}, targets = {targets.unsqueeze(-1).view(-1)}, batch_loss = {loss}")
    # Calcolo della loss media e dell'accuracy totale sul training set
    avg_loss = total_loss / len(dataloader)
    precision = precision_score(all_targets, all_preds, average=metrics_average_mode)
    recall = recall_score(all_targets, all_preds, average=metrics_average_mode)
    f1 = f1_score(all_targets, all_preds, average=metrics_average_mode)
    try:
        conf_matrix = confusion_matrix(np.argmax(all_targets, axis=1), np.argmax(all_preds, axis=1))
        # print("Training Confusion Matrix")
        # print(conf_matrix)
    except:
        conf_matrix = None
        
    # Class-wise metrics (precision, recall, f1 score)
    print(classification_report(all_targets, all_preds, target_names=PATHWAYS.keys()))

    # Save the model
    if save_all_models:
        try:
            torch.save(model.state_dict(), os.path.join(EXPERIMENT_FOLDER, "pt", f"train_{epoch_n}_{model.__class__.__name__}.pt"))
        except:
            print("Error saving model")
    
    if not return_model:
        return avg_loss, precision, recall, f1, conf_matrix
    else:
        return avg_loss, precision, recall, f1, conf_matrix, model
