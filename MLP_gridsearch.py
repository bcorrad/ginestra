import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import product

from alternative_dataset_builder import mlp_train_dataloader, mlp_val_dataloader, mlp_test_dataloader
from config import GRID_N_EPOCHS, LABELS_CODES, EXPERIMENT_FOLDER, N_RUNS
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
    
    # Initialize the lists for storing the results for each run in N_RUNS
    GRID_TRAIN_LOSS = []
    GRID_TRAIN_PRECISION = []
    GRID_TRAIN_RECALL = []
    GRID_TRAIN_F1 = []

    GRID_VAL_LOSS = []
    GRID_VAL_PRECISION = []
    GRID_VAL_RECALL = []
    GRID_VAL_F1 = []

    for config_idx, values in enumerate(product(*param_grid.values())):
        log_string = f"Testing configuration {config_idx + 1}/{len(list(product(*param_grid.values())))}"
        print(log_string)
        with open(os.path.join(EXPERIMENT_FOLDER, "log.txt"), "a") as f:
            f.write(log_string + "\n")
            
        # Initialize a report file to save in EXPERIMENT_FOLDER
        report_file = os.path.join(EXPERIMENT_FOLDER, "reports", f"report_grid_MLP_{config_idx}.txt")
        
        mlp_config = dict(zip(param_grid.keys(), values))
        print("Testing config:", mlp_config)
        
        with open(report_file, "a") as f:
            # Write the current configuration to the report file
            f.write(f"Testing config: {mlp_config}\n")
            
        for run in range(N_RUNS):
            print(f"Run {run+1}/{N_RUNS}")
            # Initialize the model, optimizer, and loss function
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
            criterion = nn.BCEWithLogitsLoss()

            for epoch in range(GRID_N_EPOCHS):            
                train_loss, precision, recall, f1, _ = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
                val_loss, val_precision, val_recall, val_f1, _ = evaluate(model, val_loader, device, criterion, epoch)
                
                GRID_TRAIN_LOSS.append(train_loss)
                GRID_TRAIN_PRECISION.append(precision)
                GRID_TRAIN_RECALL.append(recall)
                GRID_TRAIN_F1.append(f1)
                GRID_VAL_LOSS.append(val_loss)
                GRID_VAL_PRECISION.append(val_precision)
                GRID_VAL_RECALL.append(val_recall)
                GRID_VAL_F1.append(val_f1)

                training_log = f"[TRAINING RUN {run+1}/{N_RUNS} EPOCH {epoch+1}/{GRID_N_EPOCHS}] Train Loss: {train_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
                validation_log = f"[VALIDATION RUN {run+1}/{N_RUNS} EPOCH {epoch+1}/{GRID_N_EPOCHS}] Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}"
                print(training_log)
                print(validation_log)

                with open(report_file, "a") as f:
                    f.write(training_log + "\n")
                    f.write(validation_log + "\n")

        # Average the results over N_RUNS
        avg_train_loss = sum(GRID_TRAIN_LOSS) / len(GRID_TRAIN_LOSS)
        avg_train_precision = sum(GRID_TRAIN_PRECISION) / len(GRID_TRAIN_PRECISION)
        avg_train_recall = sum(GRID_TRAIN_RECALL) / len(GRID_TRAIN_RECALL)
        avg_train_f1 = sum(GRID_TRAIN_F1) / len(GRID_TRAIN_F1)
        avg_val_loss = sum(GRID_VAL_LOSS) / len(GRID_VAL_LOSS)
        avg_val_precision = sum(GRID_VAL_PRECISION) / len(GRID_VAL_PRECISION)
        avg_val_recall = sum(GRID_VAL_RECALL) / len(GRID_VAL_RECALL)
        avg_val_f1 = sum(GRID_VAL_F1) / len(GRID_VAL_F1)
        print(f"Results on {N_RUNS} runs:")
        print(f"Average Training Loss: {avg_train_loss:.4f}, Precision: {avg_train_precision:.4f}, Recall: {avg_train_recall:.4f}, F1: {avg_train_f1:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}, Precision: {avg_val_precision:.4f}, Recall: {avg_val_recall:.4f}, F1: {avg_val_f1:.4f}")
        with open(report_file, "a") as f:
            f.write(f"Results on {N_RUNS} runs:\n")
            f.write(f"Average Training Loss: {avg_train_loss:.4f}, Precision: {avg_train_precision:.4f}, Recall: {avg_train_recall:.4f}, F1: {avg_train_f1:.4f}\n")
            f.write(f"Average Validation Loss: {avg_val_loss:.4f}, Precision: {avg_val_precision:.4f}, Recall: {avg_val_recall:.4f}, F1: {avg_val_f1:.4f}\n")

        if avg_val_loss < best_score:
            best_score = avg_val_loss
            best_params = mlp_config
            
        # Empty the lists for the next run
        GRID_TRAIN_LOSS = []
        GRID_TRAIN_PRECISION = []
        GRID_TRAIN_RECALL = []
        GRID_TRAIN_F1 = []
        GRID_VAL_LOSS = []
        GRID_VAL_PRECISION = []
        GRID_VAL_RECALL = []
        GRID_VAL_F1 = []

    print("\nBest Config:", best_params)
    print(f"Best Loss: {best_score:.4f}")
    with open(os.path.join(EXPERIMENT_FOLDER, "best_config.txt"), "w") as f:
        f.write(f"Best Config: {best_params}\n")
        f.write(f"Best Loss: {best_score:.4f}\n")

# Run the search
if __name__ == "__main__":

    train_dataloader = mlp_train_dataloader
    val_dataloader = mlp_val_dataloader
    test_dataloader = mlp_test_dataloader
    N_FEATURES = len(train_dataloader.dataset[0][1][0])
    grid_search(train_dataloader, val_dataloader, test_dataloader, N_FEATURES)
