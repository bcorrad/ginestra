import os, time, requests
import torch
import torch.nn as nn
import torch.optim as optim

from utils.seed import set_seed
from utils.earlystop import EarlyStopping
from utils.experiment_init import initialize_experiment
from utils.optuna_plots import optuna_plot
from utils.print_stats import final_stats
from utils.epoch_functions import training_epoch, evaluation_epoch
            
from config import TOKEN, CHAT_ID, USERNAME, DEVICE as device
from utils.send_telegram_message import send_telegram_message

from models.MLP import *

import optuna, wandb
import optuna.samplers as samplers

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from gridsearch_dataset_builder import prepare_dataloaders
from config import GRID_N_EPOCHS, N_RUNS, LABELS_CODES, TARGET_TYPE, BASEDIR, DATASET_ID, EARLY_PATIENCE, EARLY_MIN_DELTA, USE_MULTILABEL

MODEL_NAME = "mlp"

train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders("mlp")

EXPERIMENT_FOLDER = initialize_experiment(f"{MODEL_NAME.upper()}_{DATASET_ID}", TARGET_TYPE, BASEDIR)

wandb_kwargs = {"project": EXPERIMENT_FOLDER.split('/')[-1],
                "notes": f"{EXPERIMENT_FOLDER.split('/')[-1].split('_')[0].upper()} model for {TARGET_TYPE} classification on {DATASET_ID} dataset",
                "dir": os.path.join(EXPERIMENT_FOLDER, "wandb")}
wandb.init(**wandb_kwargs)
wandb.run._redirect = False

PARAM_GRID = {
    'unit1': [3072, 4608],
    'unit2': [2304, 1536],
    'unit3': [1152, 768],
    'drop_rate': [0.1],
    'learning_rate': [1e-5],
    'l2_rate': [1e-6],
}

def objective(trial, train_loader, val_loader, test_loader, num_classes, config_idx, n_config):
    
    grid_config = {
        'unit1': trial.suggest_categorical("unit1", PARAM_GRID['unit1']),
        'unit2': trial.suggest_categorical("unit2", PARAM_GRID['unit2']),
        'unit3': trial.suggest_categorical("unit3", PARAM_GRID['unit3']),
        'drop_rate': trial.suggest_categorical("drop_rate", PARAM_GRID['drop_rate']),
        'learning_rate': trial.suggest_categorical("learning_rate", PARAM_GRID['learning_rate']),
        'l2_rate': trial.suggest_categorical("l2_rate", PARAM_GRID['l2_rate']),
    }
    
    curr_config_report_file = os.path.join(EXPERIMENT_FOLDER, "reports", f"report_optuna_{MODEL_NAME}_{config_idx}.txt")
    grid_statistics = {
        'train_loss': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'val_loss': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'epoch_time': []
    }

    for run in range(N_RUNS):
        wandb_run = wandb.init(
            project=wandb_kwargs["project"],
            name=f"CONFIG_{config_idx}_RUN_{run+1}",
            dir=wandb_kwargs["dir"],
            group=f"CONFIG_{config_idx}"
        )

        wandb_config = {
            'unit1': grid_config['unit1'],
            'unit2': grid_config['unit2'],
            'unit3': grid_config['unit3'],
            'drop_rate': grid_config['drop_rate'],
            'learning_rate': grid_config['learning_rate'],
            'l2_rate': grid_config['l2_rate'],
            'n_epochs': GRID_N_EPOCHS,
            'n_runs': N_RUNS,
            'target_type': TARGET_TYPE,
            'dataset_id': DATASET_ID,
            'run': run + 1,
            'config_idx': config_idx
        }

        wandb_run.config.update(wandb_config)
        wandb_run.log(wandb_config)

        set_seed(run + 42)
        model = MLP(
            unit1=grid_config['unit1'],
            unit2=grid_config['unit2'],
            unit3=grid_config['unit3'],
            drop_rate=grid_config['drop_rate'],
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
        print(f"Model initialized with config: {grid_config}")
        # Save all to text file
        with open(curr_config_report_file, "a") as f:
            f.write(f"Configuration {config_idx}/{n_config} - Run {run+1}/{N_RUNS}\n")
            f.write(f"Number of parameters: {num_params}\n")
            f.write(f"Model summary: {model}\n")
            f.write(f"Model configuration: {grid_config}\n")

        optimizer = optim.Adam(model.parameters(), lr=grid_config['learning_rate'], weight_decay=grid_config['l2_rate'])
        criterion = nn.CrossEntropyLoss() if not USE_MULTILABEL else nn.BCEWithLogitsLoss()
        
        early_stopping = EarlyStopping(
            patience=EARLY_PATIENCE,
            min_delta=EARLY_MIN_DELTA,
            verbose=True,
            path=os.path.join(EXPERIMENT_FOLDER, "models", f"best_model_config_{config_idx}_run_{run+1}.pt")
        )

        for epoch in range(GRID_N_EPOCHS):
            start_time = time.time()
            train_loss, train_precision, train_recall, train_f1 = training_epoch(model, train_loader, optimizer, criterion, device)
            end_time = time.time()

            print(grid_config)
            
            log_train = f"[CONFIG {config_idx}/{n_config}][{MODEL_NAME.upper()} TRAINING RUN {run+1}/{N_RUNS} EPOCH {epoch+1}/{GRID_N_EPOCHS}] Train Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Epoch Time: {end_time - start_time:.2f} seconds"
            print(log_train)
            
            val_loss, val_precision, val_recall, val_f1, topk = evaluation_epoch(model, val_loader, criterion, device)
            log_val = f"[CONFIG {config_idx}/{n_config}][{MODEL_NAME.upper()} VALIDATION RUN {run+1}/{N_RUNS} EPOCH {epoch+1}/{GRID_N_EPOCHS}] Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Top-1 Accuracy: {topk['1']:.4f}, Top-3 Accuracy: {topk['3']:.4f}, Top-5 Accuracy: {topk['5']:.4f}, Top-1 Coverage: {topk['1_coverage']:.4f}, Top-3 Coverage: {topk['3_coverage']:.4f}, Top-5 Coverage: {topk['5_coverage']:.4f}"
            print(log_val)
            
            telegram_log = f"<b>== {USERNAME} == {EXPERIMENT_FOLDER.split('/')[-1]}\n{TARGET_TYPE.upper()}</b>" + '\n <b>Training</b>\n' + log_train + '\n <b>Validation</b>\n' + log_val
            send_telegram_message(telegram_log, TOKEN, CHAT_ID) if epoch % 10 == 0 else None
            
            # Write to current config report file
            with open(curr_config_report_file, "a") as f:
                f.write(log_train + "\n")
                f.write(log_val + "\n")
            
            wandb_run.log({
                'train_loss': train_loss,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'epoch_time': end_time - start_time
            })
            
            grid_statistics['train_loss'].append(train_loss)
            grid_statistics['train_precision'].append(train_precision)
            grid_statistics['train_recall'].append(train_recall)
            grid_statistics['train_f1'].append(train_f1)
            grid_statistics['val_loss'].append(val_loss)
            grid_statistics['val_precision'].append(val_precision)
            grid_statistics['val_recall'].append(val_recall)
            grid_statistics['val_f1'].append(val_f1)
            grid_statistics['epoch_time'].append(end_time - start_time)
            
            # EarlyStopping step
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Stopped early at epoch {epoch}")
                # Load the best model and test it
                model.load_state_dict(torch.load(early_stopping.path))
                test_loss, test_precision, test_recall, test_f1, test_topk = evaluation_epoch(model, test_loader, criterion, device)
                log_test = f"[CONFIG {config_idx}/{n_config}][{MODEL_NAME.upper()} TESTING RUN {run+1}/{N_RUNS}] Test Loss: {test_loss:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, Top-1 Accuracy: {test_topk['1']:.4f}, Top-3 Accuracy: {test_topk['3']:.4f}, Top-5 Accuracy: {test_topk['5']:.4f}, Top-1 Coverage: {test_topk['1_coverage']:.4f}, Top-3 Coverage: {test_topk['3_coverage']:.4f}, Top-5 Coverage: {test_topk['5_coverage']:.4f}"
                print(log_test)
                telegram_log = f"<b>== {USERNAME} == {EXPERIMENT_FOLDER.split('/')[-1]}\n{TARGET_TYPE.upper()}</b>" + '\n <b>Testing</b>\n' + log_test
                send_telegram_message(telegram_log, TOKEN, CHAT_ID)
                with open(curr_config_report_file, "a") as f:
                    f.write(log_test + "\n")
                wandb_run.log({
                    'test_loss': test_loss,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1
                })
                wandb_run.finish()
                break
            
        # Print statistics
        final_stats(grid_statistics, config_idx, n_config)
        wandb_run.finish()

    return torch.mean(torch.tensor(grid_statistics['val_loss'])), torch.mean(torch.tensor(grid_statistics['val_f1']))

def export_results_to_csv(study, filename='optuna_results.csv'):
    df = study.trials_dataframe()
    df.to_csv(filename, index=False)
    print(f"Risultati esportati in {filename}")


def optuna_grid_search(train_loader, val_loader, test_loader, num_classes):
    param_grid = PARAM_GRID
    search_space = {key: list(values) for key, values in param_grid.items()}
    sampler = samplers.GridSampler(search_space)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def wrapped_objective(trial):
        config_idx = len(study.trials)
        n_config = len(sampler._all_grids)
        return objective(trial, train_loader, val_loader, test_loader, num_classes, config_idx, n_config)

    study.optimize(wrapped_objective, n_trials=len(sampler._all_grids))
    
    optuna_plot(study, os.path.join(EXPERIMENT_FOLDER, 'optuna_plot.png'))
    
    best = study.best_trial

    return best.params, study


if __name__ == "__main__":
    train_dataloader = train_dataloader
    val_dataloader = val_dataloader
    test_dataloader = test_dataloader
    sample = next(iter(train_dataloader))
    num_classes = len(LABELS_CODES)

    best_params, study = optuna_grid_search(train_dataloader, val_dataloader, test_dataloader, num_classes)
    export_results_to_csv(study, os.path.join(EXPERIMENT_FOLDER, 'optuna_results_{MODEL_NAME}.csv'))
    
    from utils.reports_scraper import process_all_experiments
    process_all_experiments(EXPERIMENT_FOLDER)
    print("All experiments processed.")
