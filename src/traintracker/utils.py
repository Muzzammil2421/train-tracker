import os
import re
from datetime import datetime

import pandas as pd
import torch


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


def latest_weights_path(path):
    files = os.listdir(path)
    train_files = []
    pattern = '.pt$'

    for file_name in files:
        match = re.search(pattern, file_name)
        if match:
            train_files.append(file_name)
    if len(train_files)>0:
        train_files.sort()
        last_weight_file_name = train_files[-1]
        return f"{path}/{last_weight_file_name}"
    return None
