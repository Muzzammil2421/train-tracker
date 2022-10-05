from datetime import datetime
from time import time

import pandas as pd
import os

import torch


def save_hpyerparameters(model, saving_path, batch_size, optimizer, loss_function,
                         extra_info: dict = None):
    if extra_info is not None:
        extra_info = dict(sorted(extra_info.items()))

    last_epochs_idx = __get_last_epoch_idx(saving_path) + 1

    current_data = {"epoch idx from": last_epochs_idx, "model architecture": str(model),
                    "train batch size": batch_size,
                    "loss function": str(loss_function), "optimizer": str(optimizer), "changed hyperParameters": []}
    last_row_data = __train_data_last_row(saving_path)

    for key, value in extra_info.items():
        current_data[key] = value

    if last_row_data is not None:
        current_data["changed hyperParameters"] = different_keys(current_data, last_row_data)

    if last_row_data is None or len(current_data["changed hyperParameters"]) > 0:
        df = pd.DataFrame(data=[current_data.values()], columns=current_data.keys())
        file_path = f"{saving_path}/train_data.csv"
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, index=False, header=False, mode="a")


def different_keys(current_data, past_data) -> list:
    different_keys_lst = []
    for key, value in current_data.items():
        if key == "epoch idx from" or key == "changed hyperParameters":
            continue
        if key not in past_data or current_data[key] != past_data[key]:
            different_keys_lst.append(key)
    return different_keys_lst


def __train_data_last_row(saving_path):
    last_row_dict = {}
    if len(saving_path) == 0:
        full_path = "train_data.csv"
    else:
        full_path = f"{saving_path}/train_data.csv"
    if not os.path.exists(full_path):
        return None
    df = pd.read_csv(full_path)
    if df is None or len(df) == 0:
        return None
    for attr_name, attr_value in df.iloc[-1].iteritems():
        last_row_dict[attr_name] = attr_value

    if len(last_row_dict) == 0:
        return None
    return last_row_dict


def __get_last_epoch_idx(saving_path):
    if saving_path is None or len(saving_path) == 0:
        full_path = "epochs_data.csv"
    else:
        full_path = f"{saving_path}/epochs_data.csv"

    last_epoch_idx = 0
    if os.path.exists(full_path):
        df = pd.read_csv(full_path, header=False, index_col=False)
        if df is not None:
            last_epoch_idx = len(df)
    return last_epoch_idx


def save_epoch_to_csv(saving_path, train_loss, no_train_rows, test_loss, no_test_rows, time_taken,
                      ):
    date_now = datetime.now()
    if saving_path is None or len(saving_path) == 0:
        full_path = "epochs_data.csv"
    else:
        full_path = f"{saving_path}/epochs_data.csv"
    row = [
        [train_loss, no_train_rows, test_loss, no_test_rows, time_taken, date_now.strftime('%d/%m/%Y'),
         date_now.strftime('%H:%M:00')]]
    df = pd.DataFrame(row,
                      columns=["Train Loss", "no train rows", "Test Loss", "No test rows", "Time taken (M)",
                               "Date", "Time"])

    if not os.path.exists(full_path):
        df.to_csv(full_path, index=False)
    else:
        df.to_csv(full_path, mode='a', header=False, index=False)


def save_train_weights(model, train_loss, test_loss, saving_path):
    """
    saves model weights with file name format Day_Month Hour_minute train_(train_loss) test_(test_loss)
    :param model: model object
    :param train_loss: train loss (float)
    :param test_loss: test loss (float)
    :param saving_path: the path you want to save the weights in
    :return: the full path of the saved file (saving_path+filename)
    """
    weight_file_name = f"{datetime.now().strftime('%m_%d %H_%M')} Train_({str(train_loss)[:8]}) Test_({str(test_loss)[:8]}).pt"
    full_path = f"{saving_path}/{weight_file_name}"

    torch.save(model.state_dict(), full_path)
    return full_path
