import os

import numpy as np
import torch
from torch import optim

from models.losses import mape_loss, mase_loss, smape_loss
from models.tools import EarlyStopping, adjust_learning_rate

def _acquire_device(use_gpu, gpu, use_multi_gpu, devices):
    if use_gpu:
        if use_multi_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = torch.device("cpu")
    return device

def train_model(model, config, X_train, y_train, X_val, y_val, experiment_path=""):
    device = _acquire_device(config.use_gpu, config.gpu, config.use_multi_gpu, config.devices)
    # Split batches for training
    X_train = np.array_split(X_train, int(np.ceil(X_train.shape[0] / config.batch_size)), axis=0)
    y_train = np.array_split(y_train, int(np.ceil(y_train.shape[0] / config.batch_size)), axis=0)
    # Validation data is split in 1 batch, no need to make batches
    path = "models/" + experiment_path

    early_stopping = EarlyStopping(patience=config.patience, verbose=False)

    model_optim = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = smape_loss()

    for epoch in range(config.train_epochs):
        train_loss = []

        model.train()
        for batch_x, batch_y, in zip(X_train, y_train):
            model_optim.zero_grad()
            batch_x = torch.tensor(batch_x, dtype=torch.float32).float().to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.float32).float().to(device)

            outputs = model(batch_x, None, None, None)

            loss_value = criterion(batch_x, outputs, batch_y, torch.ones_like(batch_y))
            loss = loss_value
            train_loss.append(loss.item())

            loss.backward()
            model_optim.step()

        train_loss = np.average(train_loss)
        vali_loss = validate(model, config, X_val, y_val, criterion)
        # print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(epoch + 1, train_loss, vali_loss))
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            # print("Early stopping")
            break

        adjust_learning_rate(model_optim, epoch + 1, config)

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))

    return model

def validate(model, config, X_val, y_val, criterion):
    device = _acquire_device(config.use_gpu, config.gpu, config.use_multi_gpu, config.devices)
    x = torch.tensor(X_val, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x,None,None,None).detach().cpu()
        true = torch.from_numpy(y_val)
        loss = criterion(x.detach().cpu(), outputs, true, torch.ones_like(true))

    model.train()
    return loss

def predict(model, config, X_test, is_ragged_length):
    device = _acquire_device(config.use_gpu, config.gpu, config.use_multi_gpu, config.devices)
    if not is_ragged_length:
        x = torch.tensor(X_test, dtype=torch.float32).to(device)
    else:
        x = X_test

    model.eval()
    with torch.no_grad():
        outputs = model(x,None,None,None).detach().cpu()
        preds = outputs.numpy()

    model.train()
    return preds