
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.samplers import GridSampler
import pandas as pd
import os, time

from models.GATE import *
from gridsearch_dataset_builder import prepare_dataloaders

from utils.earlystop import EarlyStopping
from config import GRID_N_EPOCHS, N_RUNS, LABELS_CODES, TARGET_TYPE, BASEDIR, USE_FINGERPRINT

from utils.seed import set_seed

mlp_train_dataloader, mlp_val_dataloader, mlp_test_dataloader, gnn_train_dataloader, gnn_val_dataloader, gnn_test_dataloader = prepare_dataloaders("gate")

PARAM_GRID = {
    'hidden_channels': [16, 32, 64, 128],
    'drop_rate': [0.1, 0.2, 0.5],
    'learning_rate': [1e-3, 1e-4],
    'l2_rate': [1e-2, 1e-3],
    'n_heads': [2, 4],
}

from utils.experiment_init import initialize_experiment
EXPERIMENT_FOLDER = initialize_experiment("gate", TARGET_TYPE, BASEDIR)

def objective(trial, train_loader, val_loader, in_channels, out_channels, edge_dim, fingerprint_length, config_idx, n_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        'hidden_channels': trial.suggest_categorical("hidden_channels", PARAM_GRID['hidden_channels']),
        'drop_rate': trial.suggest_categorical("drop_rate", PARAM_GRID['drop_rate']),
        'learning_rate': trial.suggest_categorical("learning_rate", PARAM_GRID['learning_rate']),
        'l2_rate': trial.suggest_categorical("l2_rate", PARAM_GRID['l2_rate']),
        'n_heads': trial.suggest_categorical("n_heads", PARAM_GRID['n_heads']),
    }

    report_file = os.path.join(EXPERIMENT_FOLDER, "reports", f"report_optuna_GATE_{config_idx}.txt")
    with open(report_file, "a") as f:
        f.write(f"Testing config: {config}\n")

    GRID_TRAIN_LOSS, GRID_TRAIN_PRECISION, GRID_TRAIN_RECALL, GRID_TRAIN_F1 = [], [], [], []
    GRID_VAL_LOSS, GRID_VAL_PRECISION, GRID_VAL_RECALL, GRID_VAL_F1 = [], [], [], []
    GRID_TOPK_ACCURACY_1, GRID_TOPK_ACCURACY_3, GRID_TOPK_ACCURACY_5 = [], [], []

    for run in range(N_RUNS):
        model = GATE(
            in_channels=in_channels,
            hidden_channels=config['hidden_channels'],
            out_channels=out_channels,
            edge_dim=edge_dim,
            n_heads=config['n_heads'],
            fingerprint_length=fingerprint_length
        ).to(device)

        # Reset the model weights
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # Print the model architecture
        print(model)
        # Print the number of parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params}")
        # Print the model summary
        print(f"Model summary: {model}")
        # Print the model configuration
        print(f"Model initialized with config: {config}")
        # Save all to text file
        with open(report_file, "a") as f:
            f.write(f"Configuration {config_idx}/{n_config} - Run {run+1}/{N_RUNS}\n")
            f.write(f"Number of parameters: {num_params}\n")
            f.write(f"Model summary: {model}\n")
            f.write(f"Model configuration: {config}\n")

        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2_rate'])
        criterion = nn.BCEWithLogitsLoss()
        early_stopper = EarlyStopping(patience=5 if TARGET_TYPE == "class" else 10, min_delta=0.001)

        for epoch in range(GRID_N_EPOCHS):
            start_time = time.time()
            train_loss, precision, recall, f1, _, train_model = train_epoch(model, train_loader, optimizer, criterion, device, str(epoch), return_model=True, save_all_models=False)
            end_time = time.time()
            print(f"Epoch {epoch+1}/{GRID_N_EPOCHS} took {end_time - start_time:.2f} seconds")
            val_loss, val_precision, val_recall, val_f1, _, val_model, val_topk_accuracy = evaluate(model, val_loader, device, criterion, str(epoch), return_model=True, save_all_models=False)

            GRID_TRAIN_LOSS.append(train_loss)
            GRID_TRAIN_PRECISION.append(precision)
            GRID_TRAIN_RECALL.append(recall)
            GRID_TRAIN_F1.append(f1)
            GRID_VAL_LOSS.append(val_loss)
            GRID_VAL_PRECISION.append(val_precision)
            GRID_VAL_RECALL.append(val_recall)
            GRID_VAL_F1.append(val_f1)
            GRID_TOPK_ACCURACY_1.append(val_topk_accuracy["top_1"])
            GRID_TOPK_ACCURACY_3.append(val_topk_accuracy["top_3"])
            GRID_TOPK_ACCURACY_5.append(val_topk_accuracy["top_5"])

            log_train = f"[CONFIG {config_idx}/{n_config}][GATE TRAINING RUN {run+1}/{N_RUNS} EPOCH {epoch+1}/{GRID_N_EPOCHS}] Train Loss: {train_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
            log_val = f"[CONFIG {config_idx}/{n_config}][GATE VALIDATION RUN {run+1}/{N_RUNS} EPOCH {epoch+1}/{GRID_N_EPOCHS}] Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Top-1: {val_topk_accuracy['top_1']:.4f}, Top-3: {val_topk_accuracy['top_3']:.4f}, Top-5: {val_topk_accuracy['top_5']:.4f}"
            print(log_train)
            print(log_val)
            with open(report_file, "a") as f:
                f.write(log_train + "\n")
                f.write(log_val + "\n")

            if early_stopper(val_loss) or (val_precision < 0.15 and epoch > 10 and epoch % 5 == 0):
                print(f"[EARLY STOPPING at epoch {epoch+1}] No improvement in {early_stopper.patience} epochs.")
                # Save the early stopping model
                try:
                    torch.save(train_model.state_dict(), os.path.join(EXPERIMENT_FOLDER, "pt", f"C-{config_idx}_E-{epoch}_train_early_stopping_model.pth"))
                    torch.save(val_model.state_dict(), os.path.join(EXPERIMENT_FOLDER, "pt", f"C-{config_idx}_E-{epoch}_val_early_stopping_model.pth"))
                except Exception as e:
                    print(f"Error saving model: {e}")
                with open(report_file, "a") as f:
                    f.write(f"[EARLY STOPPING at epoch {epoch+1}]\n")
                break

    # Final summary stats
    avg_train_loss = sum(GRID_TRAIN_LOSS) / len(GRID_TRAIN_LOSS)
    avg_train_precision = sum(GRID_TRAIN_PRECISION) / len(GRID_TRAIN_PRECISION)
    avg_train_recall = sum(GRID_TRAIN_RECALL) / len(GRID_TRAIN_RECALL)
    avg_train_f1 = sum(GRID_TRAIN_F1) / len(GRID_TRAIN_F1)
    std_train_loss = torch.std(torch.tensor(GRID_TRAIN_LOSS))
    std_train_precision = torch.std(torch.tensor(GRID_TRAIN_PRECISION))
    std_train_recall = torch.std(torch.tensor(GRID_TRAIN_RECALL))
    std_train_f1 = torch.std(torch.tensor(GRID_TRAIN_F1))

    avg_val_loss = sum(GRID_VAL_LOSS) / len(GRID_VAL_LOSS)
    avg_val_precision = sum(GRID_VAL_PRECISION) / len(GRID_VAL_PRECISION)
    avg_val_recall = sum(GRID_VAL_RECALL) / len(GRID_VAL_RECALL)
    avg_val_f1 = sum(GRID_VAL_F1) / len(GRID_VAL_F1)
    std_val_loss = torch.std(torch.tensor(GRID_VAL_LOSS))
    std_val_precision = torch.std(torch.tensor(GRID_VAL_PRECISION))
    std_val_recall = torch.std(torch.tensor(GRID_VAL_RECALL))
    std_val_f1 = torch.std(torch.tensor(GRID_VAL_F1))
    
    avg_val_top_k_accuracy_1 = sum(GRID_TOPK_ACCURACY_1) / len(GRID_TOPK_ACCURACY_1)
    avg_val_top_k_accuracy_3 = sum(GRID_TOPK_ACCURACY_3) / len(GRID_TOPK_ACCURACY_3)
    avg_val_top_k_accuracy_5 = sum(GRID_TOPK_ACCURACY_5) / len(GRID_TOPK_ACCURACY_5)
    std_val_top_k_accuracy_1 = torch.std(torch.tensor(GRID_TOPK_ACCURACY_1))
    std_val_top_k_accuracy_3 = torch.std(torch.tensor(GRID_TOPK_ACCURACY_3))
    std_val_top_k_accuracy_5 = torch.std(torch.tensor(GRID_TOPK_ACCURACY_5))

    final_log_train = f"[CONFIG {config_idx}/{n_config}] Train Loss: {avg_train_loss:.4f} ± {std_train_loss:.4f}, Precision: {avg_train_precision:.4f} ± {std_train_precision:.4f}, Recall: {avg_train_recall:.4f} ± {std_train_recall:.4f}, F1: {avg_train_f1:.4f} ± {std_train_f1:.4f}"
    
    final_log_val = f"[CONFIG {config_idx}/{n_config}] Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}, Precision: {avg_val_precision:.4f} ± {std_val_precision:.4f}, Recall: {avg_val_recall:.4f} ± {std_val_recall:.4f}, F1: {avg_val_f1:.4f} ± {std_val_f1:.4f}"
    final_log_val += f", Top-1: {avg_val_top_k_accuracy_1:.4f} ± {std_val_top_k_accuracy_1:.4f}, Top-3: {avg_val_top_k_accuracy_3:.4f} ± {std_val_top_k_accuracy_3:.4f}, Top-5: {avg_val_top_k_accuracy_5:.4f} ± {std_val_top_k_accuracy_5:.4f}"

    print("Final Training Summary:", final_log_train)
    print("Final Validation Summary:", final_log_val)
    with open(report_file, "a") as f:
        f.write("\n" + final_log_train + "\n")
        f.write(final_log_val + "\n")
    return avg_val_loss


def export_results_to_csv(study, filename="optuna_results_gate.csv"):
    df = study.trials_dataframe()
    df.to_csv(filename, index=False)
    print(f"Risultati esportati in {filename}")


def optuna_grid_search(train_loader, val_loader, test_loader, in_channels, out_channels, edge_dim, fingerprint_length):
    param_grid = PARAM_GRID
    sampler = GridSampler(param_grid)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def wrapped_objective(trial):
        config_idx = len(study.trials)
        n_config = len(sampler._all_grids)
        print(f"Testing configuration {config_idx}/{len(sampler._all_grids)}")
        return objective(trial, train_loader, val_loader, in_channels, out_channels, edge_dim, fingerprint_length, config_idx, n_config)

    study.optimize(wrapped_objective, n_trials=len(sampler._all_grids))

    best = study.best_trial
    print("Best config:", best.params)
    print(f"Best validation loss: {best.value:.4f}")
    with open(os.path.join(EXPERIMENT_FOLDER, "best_config_gate.txt"), "w") as f:
        f.write("GATE\n")
        f.write(f"Best Config: {best.params}\n")
        f.write(f"Best Loss: {best.value:.4f}\n")
    return best.params, study


if __name__ == "__main__":
    train_dataloader = gnn_train_dataloader
    val_dataloader = gnn_val_dataloader
    test_dataloader = gnn_test_dataloader
    sample = next(iter(train_dataloader))
    in_channels = sample.x.size(-1)
    out_channels = len(LABELS_CODES)
    edge_dim = sample.edge_attr.size(-1) if hasattr(sample, "edge_attr") else None
    fingerprint_length = len(sample.fingerprint) if USE_FINGERPRINT and hasattr(sample, "fingerprint") else None

    best_params, study = optuna_grid_search(train_dataloader, val_dataloader, test_dataloader, in_channels, out_channels, edge_dim, fingerprint_length)
    export_results_to_csv(study, os.path.join(EXPERIMENT_FOLDER, "optuna_results_gate.csv"))

