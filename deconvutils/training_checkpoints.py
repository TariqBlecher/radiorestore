import numpy as np
import torch


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class RestoreBestModel:
    def __init__(self, base_path):
        self.best_epoch = None
        self.base_path = base_path
        self.min_validation_loss = np.inf

    def test_model(self, validation_loss, epoch):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.best_epoch = epoch

    def restore(self, model):
        best_epoch = torch.load(self.base_path)
        model.load_state_dict(best_epoch['model_state_dict'])
        return model


def checkpoint_best_model(test_loss, lowest_loss, model, base_path, optimizer, scheduler):
    if test_loss < lowest_loss:
        lowest_loss = test_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': test_loss,
        }, base_path)
