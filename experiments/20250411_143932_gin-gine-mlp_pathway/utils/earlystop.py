class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.train_loss = None
        self.best_model_params = None

    def early_stop(self, validation_loss: float, train_loss: float, model_state_dict: dict) -> bool:
        """
        Early stops the training.
        :param validation_loss: actual validation loss.
        :param train_loss: actual training loss.
        :param model_state_dict: dict containing actual parameters of the model.
        :return: True if validation loss doesn't get better after patience is reached, otherwise False.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.train_loss = train_loss
            self.best_model_params = {k: v.clone().detach() for k, v in model_state_dict.items()}
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False