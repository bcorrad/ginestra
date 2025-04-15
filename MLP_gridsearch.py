import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import product

from alternative_dataset_builder import mlp_train_dataloader, mlp_val_dataloader, mlp_test_dataloader
from config import N_EPOCHS, LABELS_CODES, EXPERIMENT_FOLDER
import os 

# Define the model
class MLP_GRID(nn.Module):
    def __init__(self, unit1, unit2, unit3, drop_rate, l2_rate, learning_rate, num_classes):
        super(MLP_GRID, self).__init__()
        self.fc1 = nn.Linear(2048 + 4096, 6144)
        self.bn1 = nn.BatchNorm1d(6144)
        
        self.fc2 = nn.Linear(6144, unit1)
        self.bn2 = nn.BatchNorm1d(unit1)
        
        self.fc3 = nn.Linear(unit1, unit2)
        self.bn3 = nn.BatchNorm1d(unit2)
        
        self.fc4 = nn.Linear(unit2, unit3)
        self.dropout = nn.Dropout(drop_rate)
        
        self.output = nn.Linear(unit3, num_classes)

    def forward(self, input):
        # x = torch.cat((input_f, input_b), dim=1)
        x = input
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        
        x = torch.sigmoid(self.output(x))  # Binary multi-label classification
        return x

# Hyperparameter grid
param_grid = {
    'unit1': [3072, 4608, 6144],
    'unit2': [1536, 2304, 3072],
    'unit3': [768, 1152, 1536],
    'drop_rate': [0.1, 0.2, 0.3],
    'learning_rate': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    'l2_rate': [1e-1, 1e-2, 1e-3, 1e-4],
}

# Training loop
from models.MLP import train_epoch, evaluate

# Grid search loop
def grid_search(train_loader, val_loader, test_loader, n_features):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_score = float('inf')
    best_params = None

    for config_idx, values in enumerate(product(*param_grid.values())):
        log_string = f"Testing configuration {config_idx + 1}/{len(list(product(*param_grid.values())))}"
        print(log_string)
        with open(os.path.join(EXPERIMENT_FOLDER, "log.txt"), "a") as f:
            f.write(log_string + "\n")
            
        # Initialize a report file to save in EXPERIMENT_FOLDER
        report_file = os.path.join(EXPERIMENT_FOLDER, f"report_grid_MLP_{config_idx}.txt")
        
        mlp_config = dict(zip(param_grid.keys(), values))
        print("Testing config:", mlp_config)

        model = MLP_GRID(
            unit1=mlp_config['unit1'],
            unit2=mlp_config['unit2'],
            unit3=mlp_config['unit3'],
            drop_rate=mlp_config['drop_rate'],
            l2_rate=mlp_config['l2_rate'],
            learning_rate=mlp_config['learning_rate'],
            num_classes=len(LABELS_CODES.keys())
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=mlp_config['learning_rate'], weight_decay=mlp_config['l2_rate'])
        criterion = nn.BCELoss()
        
        with open(report_file, "a") as f:
            # Write the current configuration to the report file
            f.write(f"Testing config: {mlp_config}\n")

        for epoch in range(N_EPOCHS):
            train_loss, precision, recall, f1, _ = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
            val_loss, val_precision, val_recall, val_f1, _ = evaluate(model, val_loader, device, criterion, epoch)

            print("Testing config:", mlp_config)
            training_log_string = f"[TRAINING EPOCH {epoch+1}/{N_EPOCHS}] Train Loss: {train_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
            print(training_log_string)
            validation_log_string = f"[VALIDATION EPOCH {epoch+1}/{N_EPOCHS}] Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}"
            print(validation_log_string)
            with open(report_file, "a") as f:
                f.write(training_log_string + "\n")
                f.write(validation_log_string + "\n")

        if val_loss < best_score:
            best_score = val_loss
            best_params = mlp_config

    print("\nBest Config:", best_params)
    print(f"Best Loss: {best_score:.4f}")

# Run the search
if __name__ == "__main__":

    train_dataloader = mlp_train_dataloader
    val_dataloader = mlp_val_dataloader
    test_dataloader = mlp_test_dataloader
    N_FEATURES = len(train_dataloader.dataset[0][1][0])
    grid_search(train_dataloader, val_dataloader, test_dataloader, N_FEATURES)
