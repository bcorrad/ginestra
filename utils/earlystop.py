import torch

class EarlyStopping:
    def __init__(self, 
                 patience=5, 
                 min_delta=0.0, 
                 verbose=False, 
                 path='checkpoint.pt', 
                 metric_name='val_f1'):
        """
        Args:
            patience (int): quante epoche aspettare senza miglioramento
            min_delta (float): miglioramento minimo per essere considerato valido
            verbose (bool): se True stampa messaggi
            path (str): dove salvare il miglior modello
            mode (str): 'min' per loss, 'max' per metriche tipo F1 o accuracy
            metric_name (str): nome della metrica per logging
        """
        #  if any of the following are in the metric_name, set mode to 'max'
        if any(x in metric_name for x in ['acc', 'f1', 'precision', 'recall']):
            mode = 'max'
        elif any(x in metric_name for x in ['loss', 'error']):
            mode = 'min'
        
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.mode = mode
        self.metric_name = metric_name

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.last_checkpoint_epoch = None

    def __call__(self, metric_value, model, curr_epoch):
        score = metric_value

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model, score)
            self.last_checkpoint_epoch = curr_epoch
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            self.last_checkpoint_epoch = curr_epoch
            self._save_checkpoint(model, score)
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement in {self.metric_name} for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"[EarlyStopping] Triggered. Best {self.metric_name}: {self.best_score:.4f}")
                self.early_stop = True

    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:  # 'max'
            return score > self.best_score + self.min_delta

    def _save_checkpoint(self, model, score):
        if self.verbose:
            print(f"[EarlyStopping] Saving model to {self.path} ({self.metric_name}: {score:.4f})")
        torch.save(model.state_dict(), self.path)

    def load_best(self, model):
        model.load_state_dict(torch.load(self.path))



class EarlyStoppingTraditional:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


class EarlyStoppingComplex:
    def __init__(self, train_patience=10, val_patience=15, min_delta=0.0):
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        
        self.best_train_acc = -float('inf')
        self.best_val_acc = -float('inf')
        
        self.train_patience = train_patience
        self.val_patience = val_patience

        self.loss_train_start_epoch = None
        self.loss_val_start_epoch = None
        
        self.min_delta = min_delta

        self.early_stop = False

    def __call__(self, epoch, train_loss, val_loss):
        if train_loss < self.best_train_loss - self.min_delta:
            self.best_train_loss = train_loss
            self.loss_train_start_epoch = epoch
            
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.loss_val_start_epoch = epoch

        # Check stopping conditions
        if (epoch - self.loss_train_start_epoch > self.train_patience) and (epoch - self.loss_val_start_epoch > self.val_patience):
            self.early_stop = True
            print(f"Early stopping at epoch {epoch+1}")
            return True

        return False

    def get_patience_start_epochs(self):
        return self.loss_val_start_epoch