import torch

def final_stats(statistics: dict, config_idx: int, n_config: int):
    """
    Print the final statistics of the training and validation process.
    
    Args:
        statistics (dict): A dictionary containing the training and validation statistics.
        config_idx (int): The index of the current configuration.
        n_config (int): The total number of configurations.
    """
    
    avg_train_loss = torch.mean(torch.tensor(statistics['train_loss']))
    avg_train_precision = torch.mean(torch.tensor(statistics['train_precision']))
    avg_train_recall = torch.mean(torch.tensor(statistics['train_recall']))
    avg_train_f1 = torch.mean(torch.tensor(statistics['train_f1']))
    std_train_loss = torch.std(torch.tensor(statistics['train_loss']))
    std_train_precision = torch.std(torch.tensor(statistics['train_precision']))
    std_train_recall = torch.std(torch.tensor(statistics['train_recall']))
    std_train_f1 = torch.std(torch.tensor(statistics['train_f1']))
    avg_val_loss = torch.mean(torch.tensor(statistics['val_loss']))
    avg_val_precision = torch.mean(torch.tensor(statistics['val_precision']))
    avg_val_recall = torch.mean(torch.tensor(statistics['val_recall']))
    avg_val_f1 = torch.mean(torch.tensor(statistics['val_f1']))
    std_val_loss = torch.std(torch.tensor(statistics['val_loss']))
    std_val_precision = torch.std(torch.tensor(statistics['val_precision']))
    std_val_recall = torch.std(torch.tensor(statistics['val_recall']))
    std_val_f1 = torch.std(torch.tensor(statistics['val_f1']))
    avg_epoch_time = torch.mean(torch.tensor(statistics['epoch_time']))

    final_log_train = f"[CONFIG {config_idx}/{n_config}] Train Loss: {avg_train_loss:.4f} ± {std_train_loss:.4f}, Precision: {avg_train_precision:.4f} ± {std_train_precision:.4f}, Recall: {avg_train_recall:.4f} ± {std_train_recall:.4f}, F1: {avg_train_f1:.4f} ± {std_train_f1:.4f}, Epoch Time: {avg_epoch_time:.2f} seconds"
    
    final_log_val = f"[CONFIG {config_idx}/{n_config}] Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}, Precision: {avg_val_precision:.4f} ± {std_val_precision:.4f}, Recall: {avg_val_recall:.4f} ± {std_val_recall:.4f}, F1: {avg_val_f1:.4f} ± {std_val_f1:.4f}"

    print("Final Training Summary:", final_log_train)
    print("Final Validation Summary:", final_log_val)