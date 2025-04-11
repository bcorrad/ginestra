import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
from config import TARGET_MODE, PATHWAYS
metrics_average_mode = "two_classes" if TARGET_MODE == "two_classes" else "micro"

class MLP(nn.Module):
    def __init__(self, input_channels=6144, num_categories=10):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_channels, input_channels)
        self.bn1 = nn.BatchNorm1d(input_channels)
        
        self.fc2 = nn.Linear(input_channels, 3072)
        self.bn2 = nn.BatchNorm1d(3072)
        
        self.fc3 = nn.Linear(3072, 1536)
        self.bn3 = nn.BatchNorm1d(1536)
        
        self.fc4 = nn.Linear(1536, 1536)
        self.dropout = nn.Dropout(0.2)
        
        self.fc6 = nn.Linear(1536, num_categories)
    
    def forward(self, extended_fingerprint):
        """
        Forward pass of the model.
        Args:
        extended_fingerprint: the input tensor (batch_size, num_features=6144)
        Returns:
        x: the output tensor (batch_size, num_categories)
        """
        # x = torch.cat((input_f, input_b), dim=1)  # Concatenate inputs along feature dimension        
        # x = F.relu(self.bn2(self.fc2(extended_fingerprint)))
        # x = F.relu(self.bn3(self.fc3(x)))
        # x = F.relu(self.fc4(x))
        # x = self.dropout(x)
        # x = torch.sigmoid(self.fc5(x))
        
        x = self.bn1(F.relu(self.fc1(extended_fingerprint)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc6(x)
        x = torch.sigmoid(x)
        
        return x

    
def evaluate(model, dataloader, device, criterion):
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
            # Fingerprint at index 1
            fingerprint_features = batch[1].to(device)
            if type(fingerprint_features) == list or type(fingerprint_features) == tuple:
                fingerprint_features = fingerprint_features[0]
            if len(fingerprint_features.shape) == 3:
                # Remove the batch dimension (index 1)
                fingerprint_features = fingerprint_features.squeeze(1)
            # Convert to float
            fingerprint_features = fingerprint_features.float()
            batch_samples = fingerprint_features.to(device)
            # Targets at index 2
            targets = batch[2].to(device)
            if type(targets) == list or type(targets) == tuple:
                targets = targets[0]
            if len(targets.shape) == 3:
                # Remove the batch dimension (index 1)
                targets = targets.squeeze(1)
            # Convert to float
            targets = targets.float()
            # batch_samples = batch[0].to(device)
            # # batch_labels = batch[1].to(device)
            # # Targets
            # targets = batch[1].to(device)
            # Forward pass
            out = model(batch_samples)

            # Loss
            loss = criterion(out.squeeze(-1), targets)
            total_loss += loss.item()

            # preds = torch.argmax(F.softmax(out, dim=1))
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
        print("Validation Confusion Matrix")
        print(conf_matrix)
    except:
        conf_matrix = None
        
    # Class-wise metrics (precision, recall, f1 score)
    print(classification_report(all_targets, all_preds, target_names=PATHWAYS.keys()))
    
    return avg_loss, precision, recall, f1, conf_matrix


def train_epoch(model, dataloader, optimizer, criterion, device, verbose: bool=False):
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
        # Fingerprint at index 1
        fingerprint_features = batch[1].to(device)
        if type(fingerprint_features) == list or type(fingerprint_features) == tuple:
            fingerprint_features = fingerprint_features[0]
        if len(fingerprint_features.shape) == 3:
            # Remove the batch dimension (index 1)
            fingerprint_features = fingerprint_features.squeeze(1)
        # Convert to float
        fingerprint_features = fingerprint_features.float()
        batch_samples = fingerprint_features.to(device)
        # Targets at index 2
        targets = batch[2].to(device)
        if type(targets) == list or type(targets) == tuple:
            targets = targets[0]
        if len(targets.shape) == 3:
            # Remove the batch dimension (index 1)
            targets = targets.squeeze(1)
        # Convert to float
        targets = targets.float()
        optimizer.zero_grad()
        # Forward pass
        out = model(batch_samples)
        # Compute loss and optimize
        loss = criterion(out.squeeze(-1), targets)    # CrossEntropyLoss: input=logits per ciascuna classe, labels=tensore di interi con la classe corretta per ogni campione.
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        max_idx = torch.argmax(F.softmax(out, dim=1), dim=1, keepdim=True)
        preds = torch.zeros_like(out)
        for row_idx, col_idx in enumerate(max_idx):
            preds[row_idx, col_idx] = 1
            
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
        print("Training Confusion Matrix")
        print(conf_matrix)
    except:
        conf_matrix = None
        
    # Class-wise metrics (precision, recall, f1 score)
    print(classification_report(all_targets, all_preds, target_names=PATHWAYS.keys()))
    
    return avg_loss, precision, recall, f1, conf_matrix

