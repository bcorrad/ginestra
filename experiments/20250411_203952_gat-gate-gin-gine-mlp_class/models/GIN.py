import torch
import torch.nn.functional as F
import numpy as np
from config import EXPERIMENT_FOLDER
import os
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from sklearn.metrics import classification_report

from config import PATHWAYS

from config import TARGET_MODE
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
metrics_average_mode = "two_classes" if TARGET_MODE == "two_classes" else "micro"

BCE_THRESHOLD = 0.5
class GIN(torch.nn.Module):
    """GIN"""

    def __init__(self, num_node_features, dim_h, num_classes, **kwargs):   #, num_heads=4
        super(GIN, self).__init__()
        
        if "fingerprint_length" in kwargs and kwargs["fingerprint_length"] is not None:
            self.fingerprint_processor = torch.nn.Sequential(
                                    torch.nn.Linear(kwargs["fingerprint_length"], dim_h),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(dim_h, dim_h))
        else:
            self.fingerprint_processor = None
            
        self.conv1 = GINConv(
            Sequential(Linear(num_node_features, dim_h),
                       BatchNorm1d(dim_h), 
                       ReLU(),
                       Linear(dim_h, dim_h), 
                       ReLU()))
  
        self.conv2 = GINConv(Sequential(Linear(dim_h, dim_h), 
                       BatchNorm1d(dim_h), 
                       ReLU(),
                       Linear(dim_h, dim_h), 
                       ReLU()))
        
        self.conv3 = GINConv(Sequential(Linear(dim_h, dim_h), 
                                        BatchNorm1d(dim_h), 
                                        ReLU(),
                                        Linear(dim_h, dim_h), 
                                        ReLU()))
        # Self-Attention Layer (Multi-Head)
        # self.attention = MultiheadAttention(embed_dim=dim_h, num_heads=num_heads, batch_first=True)


        # Classificatore finale
        if "fingerprint_length" not in kwargs or kwargs["fingerprint_length"] is None:
            self.lin1 = torch.nn.Linear(3*dim_h, 3*dim_h)  
            self.lin2 = torch.nn.Linear(3*dim_h, num_classes)
        else:
            self.lin1 = torch.nn.Linear(4*dim_h, 4*dim_h)
            self.lin2 = torch.nn.Linear(4*dim_h, num_classes)

        # self.lin1 = Linear(dim_h, dim_h)
        # self.lin2 = Linear(dim_h, num_classes)


    def forward(self, x, edge_index, batch, p=0.2, **kwargs):
        
        if "fingerprint" in kwargs:
            fingerprint = kwargs["fingerprint"]
            fingerprint_emb = self.fingerprint_processor(torch.Tensor(fingerprint))
        else:
            fingerprint = None

        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        # Dropout 
        h1 = F.dropout(h1, p=p, training=self.training)
        h2 = self.conv2(h1, edge_index)
        h2 = F.dropout(h2, p=p, training=self.training)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        #The authors make two important points about graph-level readout:

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


        # Stack embeddings per livello, codifica posizionale, A+H=D, e MLP sui token (output=3 token, dim_h)
        #H = torch.stack([h1, h2, h3], dim=1)  # (batch_size, 3, dim_h)
        # Apply Self-Attention tra i livelli
        #A, _ = self.attention(H, H, H)  # (batch_size, 3, dim_h)


        # Classifier
        h = self.lin1(h)
        h = h.relu()
        # h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h


def evaluate(model, dataloader, device, criterion, epoch_n):
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
            elif model.__class__.__name__ == "GATWithEdgeFeatures":
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
    
    # Calcolo della loss media e dell'accuracy totale sul validation set
    avg_loss = total_loss / len(dataloader)
    precision = precision_score(all_targets, all_preds, average=metrics_average_mode)
    recall = recall_score(all_targets, all_preds, average=metrics_average_mode)
    f1 = f1_score(all_targets, all_preds, average=metrics_average_mode)
    try:
        conf_matrix = confusion_matrix(np.argmax(all_targets, axis=1), np.argmax(all_preds, axis=1))
        # print("Validation Confusion Matrix")
        # print(conf_matrix)
    except:
        conf_matrix = None
        
    # Class-wise metrics (precision, recall, f1 score)
    print(classification_report(all_targets, all_preds, target_names=PATHWAYS.keys()))
    # Save model if needed
    torch.save(model.state_dict(), os.path.join(EXPERIMENT_FOLDER, "pt", f"eval_{epoch_n}_{model.__class__.__name__}.pt"))
    
    return avg_loss, precision, recall, f1, conf_matrix


def train_epoch(model, dataloader, optimizer, criterion, device, epoch_n, verbose:bool=False):
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
        if model.__class__.__name__ == "GIN":
            out = model(batch.x, edge_index=batch.edge_index, batch=batch.batch)
        elif model.__class__.__name__ == "GINWithEdgeFeatures":
            from config import USE_FINGERPRINT
            if USE_FINGERPRINT:
                out = model(x=batch.x, 
                edge_index=batch.edge_index, 
                edge_attr=batch.edge_attr, 
                batch=batch.batch, 
                fingerprint=batch.fingerprint)
            else:
                out = model(x=batch.x, 
                edge_index=batch.edge_index, 
                edge_attr=batch.edge_attr, 
                batch=batch.batch)
            #out = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch, fingerprint=batch.fingerprint)
        elif model.__class__.__name__ == "GATWithEdgeFeatures":
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
    torch.save(model.state_dict(), os.path.join(EXPERIMENT_FOLDER, "pt", f"train_{epoch_n}_{model.__class__.__name__}.pt"))
    
    return avg_loss, precision, recall, f1, conf_matrix
