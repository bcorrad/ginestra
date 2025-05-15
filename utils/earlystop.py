import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False, path='checkpoint.pt'):
        """
        Args:
            patience (int): quante epoche aspettare senza miglioramento
            min_delta (float): minimo miglioramento nella loss per considerarlo reale
            verbose (bool): se True stampa messaggi quando salvi
            path (str): dove salvare il miglior modello
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path

        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                if self.verbose:
                    print("EarlyStopping triggered.")
                self.early_stop = True

    def _save_checkpoint(self, model):
        if self.verbose:
            print(f"Saving model to {self.path} (val_loss: {self.best_loss:.4f})")
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
