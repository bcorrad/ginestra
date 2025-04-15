import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
import os

from models.GIN import GIN, train_epoch, evaluate
from config import GRID_N_EPOCHS, LABELS_CODES, EXPERIMENT_FOLDER, N_RUNS
from alternative_dataset_builder import gnn_train_dataloader, gnn_val_dataloader, gnn_test_dataloader

# Hyperparameter grid
param_grid = {
    'dim_h': [16, 32, 64, 128],
    'drop_rate': [0.1, 0.2],
    'learning_rate': [1e-3, 1e-4],
    'l2_rate': [1e-2, 1e-3],
}

def grid_search(train_loader, val_loader, test_loader, num_node_features, num_classes):
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

    all_configs = list(product(*param_grid.values()))
    for config_idx, values in enumerate(all_configs):
        log_string = f"Testing configuration {config_idx + 1}/{len(all_configs)}"
        print(log_string)
        with open(os.path.join(EXPERIMENT_FOLDER, "log.txt"), "a") as f:
            f.write(log_string + "\n")

        report_file = os.path.join(EXPERIMENT_FOLDER, "reports", f"report_grid_GIN_{config_idx}.txt")
        gin_config = dict(zip(param_grid.keys(), values))
        print("Testing config:", gin_config)

        with open(report_file, "a") as f:
            f.write(f"Testing config: {gin_config}\n")
            
        for run in range(N_RUNS):
            model = GIN(
                num_node_features=num_node_features,
                dim_h=gin_config['dim_h'],
                num_classes=num_classes
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=gin_config['learning_rate'], weight_decay=gin_config['l2_rate'])
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
            best_params = gin_config
            
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


if __name__ == "__main__":
    train_dataloader = gnn_train_dataloader
    val_dataloader = gnn_val_dataloader
    test_dataloader = gnn_test_dataloader
    sample = next(iter(train_dataloader))
    num_node_features = sample.x.size(-1)
    num_classes = len(LABELS_CODES)

    grid_search(train_dataloader, val_dataloader, test_dataloader, num_node_features, num_classes)
