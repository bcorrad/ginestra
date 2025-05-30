
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.samplers import GridSampler
import pandas as pd
import os

from models.GIN import GIN, train_epoch, evaluate
from config import GRID_N_EPOCHS, LABELS_CODES, EXPERIMENT_FOLDER, N_RUNS
from alternative_dataset_builder import gnn_train_dataloader, gnn_val_dataloader, gnn_test_dataloader

def objective(trial, train_loader, val_loader, test_loader, num_node_features, num_classes, config_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gin_config = {
        'dim_h': trial.suggest_categorical("dim_h", [16, 32, 64, 128]),
        'drop_rate': trial.suggest_categorical("drop_rate", [0.1, 0.2]),
        'learning_rate': trial.suggest_categorical("learning_rate", [1e-3, 1e-4]),
        'l2_rate': trial.suggest_categorical("l2_rate", [1e-2, 1e-3]),
    }

    report_file = os.path.join(EXPERIMENT_FOLDER, "reports", f"report_optuna_GIN_{config_idx}.txt")
    with open(report_file, "a") as f:
        f.write(f"Testing config: {gin_config}\n")

    GRID_TRAIN_LOSS, GRID_TRAIN_PRECISION, GRID_TRAIN_RECALL, GRID_TRAIN_F1 = [], [], [], []
    GRID_VAL_LOSS, GRID_VAL_PRECISION, GRID_VAL_RECALL, GRID_VAL_F1 = [], [], [], []

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

            log_train = f"[TRAINING RUN {run+1}/{N_RUNS} EPOCH {epoch+1}/{GRID_N_EPOCHS}] Train Loss: {train_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
            log_val = f"[VALIDATION RUN {run+1}/{N_RUNS} EPOCH {epoch+1}/{GRID_N_EPOCHS}] Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}"
            print(log_train)
            print(log_val)
            with open(report_file, "a") as f:
                f.write(log_train + "\n")
                f.write(log_val + "\n")

    avg_val_loss = sum(GRID_VAL_LOSS) / len(GRID_VAL_LOSS)
    return avg_val_loss

def test_model(best_params, train_loader, test_loader, num_node_features, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GIN(
        num_node_features=num_node_features,
        dim_h=best_params['dim_h'],
        num_classes=num_classes
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['l2_rate']
    )
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(GRID_N_EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion, device, epoch)

    test_loss, test_precision, test_recall, test_f1, _ = evaluate(model, test_loader, device, criterion, epoch)
    print(f"Test Loss: {test_loss:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    return {
        'test_loss': test_loss,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }

def export_results_to_csv(study, filename='optuna_results.csv'):
    df = study.trials_dataframe()
    df.to_csv(filename, index=False)
    print(f"Risultati esportati in {filename}")

def optuna_grid_search(train_loader, val_loader, test_loader, num_node_features, num_classes):
    param_grid = {
        'dim_h': [16, 32, 64, 128],
        'drop_rate': [0.1, 0.2],
        'learning_rate': [1e-3, 1e-4],
        'l2_rate': [1e-2, 1e-3],
    }
    search_space = {key: list(values) for key, values in param_grid.items()}
    sampler = GridSampler(search_space)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def wrapped_objective(trial):
        config_idx = len(study.trials)
        print(f"Testing configuration {config_idx + 1}/{len(sampler._all_grids)}")
        with open(os.path.join(EXPERIMENT_FOLDER, "log.txt"), "a") as f:
            f.write(f"Testing configuration {config_idx + 1}/{len(sampler._all_grids)}\n")
        return objective(trial, train_loader, val_loader, test_loader, num_node_features, num_classes, config_idx)

    study.optimize(wrapped_objective, n_trials=len(sampler._all_grids))

    best = study.best_trial
    print("\nBest Config:", best.params)
    print(f"Best Loss: {best.value:.4f}")
    with open(os.path.join(EXPERIMENT_FOLDER, "best_config.txt"), "w") as f:
        f.write(f"Best Config: {best.params}\n")
        f.write(f"Best Loss: {best.value:.4f}\n")

    return best.params, study

if __name__ == "__main__":
    train_dataloader = gnn_train_dataloader
    val_dataloader = gnn_val_dataloader
    test_dataloader = gnn_test_dataloader
    sample = next(iter(train_dataloader))
    num_node_features = sample.x.size(-1)
    num_classes = len(LABELS_CODES)

    best_params, study = optuna_grid_search(train_dataloader, val_dataloader, test_dataloader, num_node_features, num_classes)
    test_metrics = test_model(best_params, train_dataloader, test_dataloader, num_node_features, num_classes)
    export_results_to_csv(study, os.path.join(EXPERIMENT_FOLDER, 'optuna_results.csv'))
