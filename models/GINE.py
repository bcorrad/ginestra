import torch
import torch.nn.functional as F
import numpy as np
import os
from config import TARGET_MODE, PATHWAYS 

from torch_geometric.nn import global_add_pool, GINEConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from sklearn.metrics import classification_report
from utils.topk import top_k_accuracy

from config import PATHWAYS

from config import DEVICE as device
from config import TARGET_MODE
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
metrics_average_mode = "two_classes" if TARGET_MODE == "two_classes" else "micro"

BCE_THRESHOLD = 0.5

class GINE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, out_channels, **kwargs):
        super().__init__()
        
        if "fingerprint_length" in kwargs and kwargs["fingerprint_length"] is not None:
            self.fingerprint_processor = torch.nn.Sequential(
                                    torch.nn.Linear(kwargs["fingerprint_length"], hidden_channels),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_channels, hidden_channels))
        else:
            self.fingerprint_processor = None
            
        # TODO: ridurre il numero di layer dei GINEConv
        # GIN layer
        self.conv1 = GINEConv(
            Sequential(Linear(in_channels, hidden_channels), 
                            #BatchNorm1d(hidden_channels), 
                            ReLU(),
                            Linear(hidden_channels, hidden_channels), 
                            #ReLU() 
                            ),
            edge_dim=edge_dim
        )
        self.bn1 = BatchNorm1d(hidden_channels)

        self.conv2 = GINEConv(
            Sequential(Linear(hidden_channels, hidden_channels), 
                            #BatchNorm1d(hidden_channels), 
                            ReLU(),
                            Linear(hidden_channels, hidden_channels), 
                            #ReLU()
                            ),
            edge_dim=edge_dim
        )

        self.bn2 = BatchNorm1d(hidden_channels)

        # self.conv3 = GINEConv(
        #      Sequential(Linear(hidden_channels, hidden_channels), 
        #                     BatchNorm1d(hidden_channels), 
        #                     ReLU(),
        #                     Linear(hidden_channels, hidden_channels), 
        #                     ReLU()),
        #     edge_dim=edge_dim
        # )
        self.conv3 = GINEConv(
             Sequential(Linear(hidden_channels, 512), 
                            #BatchNorm1d(hidden_channels), 
                            ReLU(),
                            Linear(512, 512), 
                            #ReLU()
                            ),
            edge_dim=edge_dim
        )
        self.bn3 = BatchNorm1d(512)

        # Dropout
        if "drop_rate" in kwargs and kwargs["drop_rate"] is not None:
            self.dropout = kwargs["drop_rate"]
        else:
            self.dropout = 0.1
            
        print(f"[DROPOUT SET] Dropout: {self.dropout}")

        readout_dim = hidden_channels + hidden_channels + 512  # h1 + h2 + h3
        self.lin1 = torch.nn.Linear(readout_dim, 1024)
        self.lin2 = torch.nn.Linear(1024, out_channels)
            
        # # Classificatore finale
        # if "fingerprint_length" not in kwargs or kwargs["fingerprint_length"] is None:
        #     self.fc1 = torch.nn.Linear(3*hidden_channels, 3*hidden_channels)  
        #     self.fc2 = torch.nn.Linear(3*hidden_channels, out_channels)
        # else:
        #     self.fc1 = torch.nn.Linear(4*hidden_channels, 4*hidden_channels)
        #     self.fc2 = torch.nn.Linear(4*hidden_channels, out_channels)

        # Self Attention Layer
        # self.attention = MultiheadAttention(embed_dim=hidden_channels, num_heads=4, batch_first=True)
        # Put to cuda
        self.to(device)

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):   #p=0.2, nonlinear=False,

        if "fingerprint" in kwargs:
            fingerprint = kwargs["fingerprint"]
            fingerprint_emb = self.fingerprint_processor(torch.Tensor(fingerprint))
        else:
            fingerprint = None

        # Forward of a GINE layer, with dropout, batchnorm, and ReLU. 
        # Apply global pooling after each layer.
        h1 = self.conv1(x, edge_index, edge_attr)  # Usa x, non h1
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        # if nonlinear:
        #     h1 = F.relu(h1)

        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        # if nonlinear:
        #     h2 = F.relu(h2)

        h3 = self.conv3(h2, edge_index, edge_attr)
        h3 = self.bn3(h3)
        h3 = F.relu(h3)
        # h3 = F.dropout(h3, p=0.5, training=self.training)
        # if nonlinear:
        #     h3 = F.relu(h3)

        # Global pooling on node features
        h1_pool = global_add_pool(h1, batch)
        h2_pool = global_add_pool(h2, batch)
        h3_pool = global_add_pool(h3, batch)

        # Concatenate the embeddings and the fingerprint if not None
        if fingerprint is not None:
            h = torch.cat([h1_pool, h2_pool, h3_pool, fingerprint_emb], dim=1)
        else:
            h = torch.cat([h1_pool, h2_pool, h3_pool], dim=1)

        # # Stack embeddings per livello
        # H = torch.stack([h1_pool, h2_pool, h3_pool], dim=1)  # (batch_size, 3, dim_h)
        # # Apply Self-Attention tra i livelli
        # H, _ = self.attention(H, H, H)  # (batch_size, 3, dim_h)
        # # Pooling sulle rappresentazioni trasformate (media tra i 3 livelli)
        # h = H.mean(dim=1)  # (batch_size, dim_h)    

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = self.lin2(h)

        return h


def evaluate(model, dataloader, device, criterion, epoch_n, return_model=False, save_all_models=False, experiment_folder=None):
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
    EXPERIMENT_FOLDER = experiment_folder if experiment_folder is not None else os.path.join(os.getcwd(), "experiments")
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            if model.__class__.__name__ == "GIN":
                out = model(batch.x, edge_index=batch.edge_index, batch=batch.batch)
            elif model.__class__.__name__ == "GINE" or model.__class__.__name__ == "NewGINE":
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
            print(f"Error saving model")
            
    if not return_model:
        return avg_loss, precision, recall, f1, conf_matrix, top_k_accuracy_dict
    else:
        return avg_loss, precision, recall, f1, conf_matrix, model, top_k_accuracy_dict


def train_epoch(model, dataloader, optimizer, criterion, device, epoch_n, verbose:bool=False, return_model=False, save_all_models=False, experiment_folder=None):
    """
    Training loop for the model.
    Args:
    model: the model to train
    dataloader: the training dataloader
    optimizer: the optimizer to use
    criterion: the loss function
    device: the device to use (cpu or cuda)
    epoch_n: the current epoch number

    Returns:
    avg_loss: the average loss over the training set
    loss: the loss of the last batch

    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    EXPERIMENT_FOLDER = experiment_folder if experiment_folder is not None else os.path.join(os.getcwd(), "experiments")

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        # Forward pass
        if model.__class__.__name__ == "GIN":
            out = model(batch.x, edge_index=batch.edge_index, batch=batch.batch)
        elif model.__class__.__name__ == "GINE":
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
    
    if save_all_models:
        try:
            torch.save(model.state_dict(), os.path.join(EXPERIMENT_FOLDER, "pt", f"train_{epoch_n}_{model.__class__.__name__}.pt"))
        except:
            print(f"Error saving model")
            
    if not return_model:
        return avg_loss, precision, recall, f1, conf_matrix
    else:
        return avg_loss, precision, recall, f1, conf_matrix, model
