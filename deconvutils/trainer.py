import numpy as np
import torch
from torch import nn
from .training_checkpoints import *
from .utils import *
import logging
import torchvision
import sys

logger = logging.getLogger(__name__)


def loss_fn_l1(output, target):
    l1loss = nn.functional.l1_loss(output, target)
    return l1loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def trainer(train_loader, test_loader, model, loss_fn, optimizer, scheduler,
            base_path, epochs, crop_target=False, device='cuda'):

    train_losses = []
    test_losses = []
    lr = []

    early_stopper = EarlyStopper(patience=10, min_delta=0)
    best_model_restore = RestoreBestModel(base_path)

    model = model.to(device=device)
    lowest_loss = np.inf
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}\n-------------------------------")
        lr.append(get_lr(optimizer))
        train_losses = train_loop(train_loader, model, loss_fn, optimizer, train_losses, crop_target=crop_target, device=device)
        test_loss, test_losses = val_loop(test_loader, model, loss_fn, test_losses, crop_target=crop_target, device=device)
        if scheduler:
            scheduler.step(test_loss)
        checkpoint_best_model(test_loss=test_loss, lowest_loss=lowest_loss, model=model, base_path=base_path, optimizer=optimizer, scheduler=scheduler)

        best_model_restore.test_model(test_loss, epoch)
        if early_stopper.early_stop(test_loss):
            break
    model = best_model_restore.restore(model)

    metrics = {'train_losses': train_losses, 'test_losses': test_losses, 'lr': lr}
    torch.save(metrics, Path(Path(base_path).parent, 'metrics.pt'))

    return model, metrics


def train_loop(dataloader, model, loss_fn, optimizer, train_losses, crop_target=False, device='cuda'):
    model.train()
    size = len(dataloader.dataset)
    temp_losses = []
    for batch, (X, Y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        Y = Y.to(device)

        pred = model(X)

        if crop_target:
            Y = torchvision.transforms.functional.center_crop(Y, pred.size()[-1])

        loss = loss_fn(pred, Y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if batch % 120 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            temp_losses.append(loss)
            logger.info(f"Learning Rate {get_lr(optimizer)}")

    train_losses.append(np.mean(temp_losses))
    return train_losses


def val_loop(dataloader, model, loss_fn, test_losses, crop_target=False, device='cuda'):
    num_batches = len(dataloader)
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)
            pred = model(X)
            if crop_target:
                Y = torchvision.transforms.functional.center_crop(Y, pred.size()[-1])

            test_loss += loss_fn(pred, Y).item()

    test_loss /= num_batches
    test_losses.append(test_loss)
    logger.info(f"Test Error: Avg loss: {test_loss:>8f} \n")
    return test_loss, test_losses
