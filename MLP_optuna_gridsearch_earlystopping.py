import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.samplers import GridSampler
import os
from models.MLP import train_epoch, evaluate
import torch.nn.functional as F
from gridsearch_dataset_builder import prepare_dataloaders
import time
from utils.earlystop import EarlyStopping
from utils.seed import set_seed
from config import GRID_N_EPOCHS, N_RUNS, LABELS_CODES, TARGET_TYPE, BASEDIR


mlp_train_dataloader, mlp_val_dataloader, mlp_test_dataloader, gnn_train_dataloader, gnn_val_dataloader, gnn_test_dataloader = prepare_dataloaders("mlp", batch_size=128)

import config
from utils.experiment_init import initialize_experiment
EXPERIMENT_FOLDER = initialize_experiment(f"mlp_{config.DATASET_ID}", TARGET_TYPE, BASEDIR)

PARAM_GRID = {
    'unit1': [3072],
    'unit2': [2304],
    'unit3': [1152],
    'drop_rate': [0.1, 0.2],
    'learning_rate': [1e-5],
    'l2_rate': [1e-6],
}

# Define the model
class MLP_GRID(nn.Module):
    def __init__(self, unit1, unit2, unit3, drop_rate, num_classes):
        super(MLP_GRID, self).__init__()
        self.fc1 = nn.Linear(2048 + 4096, 6144)
        self.bn1 = nn.BatchNorm1d(6144)
        
        self.fc2 = nn.Linear(6144, unit1)
        self.bn2 = nn.BatchNorm1d(unit1)
        
        self.fc3 = nn.Linear(unit1, unit2)
        self.bn3 = nn.BatchNorm1d(unit2)
        
        self.fc4 = nn.Linear(unit2, unit3)
        self.bn4 = nn.BatchNorm1d(unit3)
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
        x = self.bn4(x)
        x = self.dropout(x)
        
        x = torch.sigmoid(self.output(x))  # Binary multi-label classification
        return x

def objective(trial, train_loader, val_loader, num_features, num_classes, config_idx, n_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlp_config = {
        'unit1': trial.suggest_categorical("unit1", PARAM_GRID['unit1']),
        'unit2': trial.suggest_categorical("unit2", PARAM_GRID['unit2']),
        'unit3': trial.suggest_categorical("unit3", PARAM_GRID['unit3']),
        'drop_rate': trial.suggest_categorical("drop_rate", PARAM_GRID['drop_rate']),
        'learning_rate': trial.suggest_categorical("learning_rate", PARAM_GRID['learning_rate']),
        'l2_rate': trial.suggest_categorical("l2_rate", PARAM_GRID['l2_rate']),
    }

    report_file = os.path.join(EXPERIMENT_FOLDER, "reports", f"report_optuna_MLP_{config_idx}.txt")
    with open(report_file, "a") as f:
        f.write(f"Testing config: {mlp_config}\n")

    GRID_TRAIN_LOSS, GRID_TRAIN_PRECISION, GRID_TRAIN_RECALL, GRID_TRAIN_F1 = [], [], [], []
    GRID_VAL_LOSS, GRID_VAL_PRECISION, GRID_VAL_RECALL, GRID_VAL_F1 = [], [], [], []
    GRID_TOPK_ACCURACY_1, GRID_TOPK_ACCURACY_3, GRID_TOPK_ACCURACY_5 = [], [], []
    EPOCH_TIMES = []

    for run in range(N_RUNS):
        set_seed(run + 42)
        # Initialize the model, optimizer, and loss function
        model = MLP_GRID(
            unit1=mlp_config['unit1'],
            unit2=mlp_config['unit2'],
            unit3=mlp_config['unit3'],
            drop_rate=mlp_config['drop_rate'],
            num_classes=num_classes
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
        print(f"Model initialized with config: {mlp_config}")
        # Save all to text file
        with open(report_file, "a") as f:
            f.write(f"Configuration {config_idx}/{n_config} - Run {run+1}/{N_RUNS}\n")
            f.write(f"Number of parameters: {num_params}\n")
            f.write(f"Model summary: {model}\n")
            f.write(f"Model configuration: {mlp_config}\n")

        optimizer = optim.Adam(model.parameters(), lr=mlp_config['learning_rate'], weight_decay=mlp_config['l2_rate'])
        criterion = nn.BCEWithLogitsLoss()

        early_stopper = EarlyStopping(patience=5 if TARGET_TYPE == "class" else 5, min_delta=0.001)

        for epoch in range(GRID_N_EPOCHS):
            start_time_train = time.time()
            train_loss, precision, recall, f1, _, train_model = train_epoch(model, train_loader, optimizer, criterion, device, str(epoch), return_model=True, save_all_models=False, experiment_folder=EXPERIMENT_FOLDER)
            end_time_train = time.time()
            start_time_val = time.time()
            val_loss, val_precision, val_recall, val_f1, _, val_model, val_topk_accuracy = evaluate(model, val_loader, device, criterion, str(epoch), return_model=True, save_all_models=False, experiment_folder=EXPERIMENT_FOLDER)
            end_time_val = time.time()


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
            EPOCH_TIMES.append(end_time_train - start_time_train)

            log_train = f"[CONFIG {config_idx}/{n_config}][MLP TRAINING RUN {run+1}/{N_RUNS} EPOCH {epoch+1}/{GRID_N_EPOCHS}] Train Loss: {train_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Epoch Time: {end_time_train - start_time_train:.2f}"
            log_val = f"[CONFIG {config_idx}/{n_config}][MLP VALIDATION RUN {run+1}/{N_RUNS} EPOCH {epoch+1}/{GRID_N_EPOCHS}] Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Top-1: {val_topk_accuracy['top_1']:.4f}, Top-3: {val_topk_accuracy['top_3']:.4f}, Top-5: {val_topk_accuracy['top_5']:.4f}"
            print(mlp_config)
            print(log_train)
            print(log_val)
            with open(report_file, "a") as f:
                f.write(log_train + "\n")
                f.write(log_val + "\n")

            if early_stopper(val_loss):
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
    avg_epoch_time = sum(EPOCH_TIMES) / len(EPOCH_TIMES)

    final_log_train = f"[CONFIG {config_idx}/{n_config}] Train Loss: {avg_train_loss:.4f} ± {std_train_loss:.4f}, Precision: {avg_train_precision:.4f} ± {std_train_precision:.4f}, Recall: {avg_train_recall:.4f} ± {std_train_recall:.4f}, F1: {avg_train_f1:.4f} ± {std_train_f1:.4f}, Epoch Time: {avg_epoch_time:.4f}"
    
    final_log_val = f"[CONFIG {config_idx}/{n_config}] Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}, Precision: {avg_val_precision:.4f} ± {std_val_precision:.4f}, Recall: {avg_val_recall:.4f} ± {std_val_recall:.4f}, F1: {avg_val_f1:.4f} ± {std_val_f1:.4f}"
    final_log_val += f", Top-1: {avg_val_top_k_accuracy_1:.4f} ± {std_val_top_k_accuracy_1:.4f}, Top-3: {avg_val_top_k_accuracy_3:.4f} ± {std_val_top_k_accuracy_3:.4f}, Top-5: {avg_val_top_k_accuracy_5:.4f} ± {std_val_top_k_accuracy_5:.4f}"

    print("Final Training Summary:", final_log_train)
    print("Final Validation Summary:", final_log_val)
    with open(report_file, "a") as f:
        f.write("\n" + final_log_train + "\n")
        f.write(final_log_val + "\n")
    return avg_val_loss


def export_results_to_csv(study, filename="optuna_results_mlp.csv"):
    df = study.trials_dataframe()
    df.to_csv(filename, index=False)
    print(f"Risultati esportati in {filename}")

def optuna_grid_search(train_loader, val_loader, test_loader, num_features, num_classes):
    # Hyperparameter grid
    param_grid = PARAM_GRID
    search_space = {key: list(values) for key, values in param_grid.items()}
    sampler = GridSampler(search_space)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def wrapped_objective(trial):
        config_idx = len(study.trials)
        n_config = len(sampler._all_grids)
        print(f"Testing configuration {config_idx}/{len(sampler._all_grids)}")
        return objective(trial, train_loader, val_loader, num_features, num_classes, config_idx, n_config)

    study.optimize(wrapped_objective, n_trials=len(sampler._all_grids))

    best = study.best_trial
    print("Best config:", best.params)
    print(f"Best validation loss: {best.value:.4f}")
    with open(os.path.join(EXPERIMENT_FOLDER, "best_config_mlp.txt"), "w") as f:
        f.write(f"MLP\n")
        f.write(f"Best Config: {best.params}\n")
        f.write(f"Best Loss: {best.value:.4f}\n")
    return best.params, study

if __name__ == "__main__":
    train_dataloader = mlp_train_dataloader
    val_dataloader = mlp_val_dataloader
    test_dataloader = mlp_test_dataloader
    num_features = len(train_dataloader.dataset[0][1][0])
    num_classes = len(LABELS_CODES)

    best_params, study = optuna_grid_search(train_dataloader, val_dataloader, test_dataloader, num_features, num_classes)
    export_results_to_csv(study, os.path.join(EXPERIMENT_FOLDER, "optuna_results_mlp.csv"))

    from utils.reports_scraper import process_all_experiments
    process_all_experiments(EXPERIMENT_FOLDER)
    print("All experiments processed.")