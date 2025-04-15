import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
import os

from models.GIN import GIN, train_epoch, evaluate
from config import N_EPOCHS, LABELS_CODES, EXPERIMENT_FOLDER
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

    all_configs = list(product(*param_grid.values()))
    for config_idx, values in enumerate(all_configs):
        log_string = f"Testing configuration {config_idx + 1}/{len(all_configs)}"
        print(log_string)
        with open(os.path.join(EXPERIMENT_FOLDER, "log.txt"), "a") as f:
            f.write(log_string + "\n")

        report_file = os.path.join(EXPERIMENT_FOLDER, f"report_grid_GIN_{config_idx}.txt")
        gin_config = dict(zip(param_grid.keys(), values))
        print("Testing config:", gin_config)

        model = GIN(
            num_node_features=num_node_features,
            dim_h=gin_config['dim_h'],
            num_classes=num_classes
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=gin_config['learning_rate'], weight_decay=gin_config['l2_rate'])
        criterion = nn.BCEWithLogitsLoss()

        with open(report_file, "a") as f:
            f.write(f"Testing config: {gin_config}\n")

        for epoch in range(N_EPOCHS):
            train_loss, precision, recall, f1, _ = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
            val_loss, val_precision, val_recall, val_f1, _ = evaluate(model, val_loader, device, criterion, epoch)

            training_log = f"[TRAINING EPOCH {epoch+1}/{N_EPOCHS}] Train Loss: {train_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
            validation_log = f"[VALIDATION EPOCH {epoch+1}/{N_EPOCHS}] Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}"
            print(training_log)
            print(validation_log)

            with open(report_file, "a") as f:
                f.write(training_log + "\n")
                f.write(validation_log + "\n")

        if val_loss < best_score:
            best_score = val_loss
            best_params = gin_config

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
