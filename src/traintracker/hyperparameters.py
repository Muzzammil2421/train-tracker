
from .utils import *


def save_hpyerparameters(model, saving_path, batch_size, optimizer, loss_function,
                         extra_info: dict = None):
    if extra_info is not None:
        extra_info = dict(sorted(extra_info.items()))

    last_epochs_idx = __get_last_epoch_idx(saving_path) + 1

    current_data = {"epoch idx from": last_epochs_idx, "model architecture": str(model),
                    "train batch size": batch_size,
                    "loss function": str(loss_function), "optimizer": str(optimizer), "changed hyperParameters": []}
    last_row_data = last_saved_hyperparameters(saving_path)

    for key, value in extra_info.items():
        current_data[key] = value

    if last_row_data is not None:
        current_data["changed hyperParameters"] = __different_keys(current_data, last_row_data)

    if last_row_data is None or len(current_data["changed hyperParameters"]) > 0:
        df = pd.DataFrame(data=[current_data.values()], columns=current_data.keys())
        file_path = f"{saving_path}/train_data.csv"
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, index=False, header=False, mode="a")


def __different_keys(current_data, past_data) -> list:
    different_keys_lst = []
    for key, value in current_data.items():
        if key == "epoch idx from" or key == "changed hyperParameters":
            continue
        if key not in past_data or current_data[key] != past_data[key]:
            different_keys_lst.append(key)
    return different_keys_lst


def last_saved_hyperparameters(saving_path) -> dict:
    last_row_dict = {}
    if len(saving_path) == 0:
        full_path = "train_data.csv"
    else:
        full_path = f"{saving_path}/train_data.csv"
    if not os.path.exists(full_path):
        return last_row_dict
    df = pd.read_csv(full_path)
    if df is None or len(df) == 0:
        return last_row_dict
    for attr_name, attr_value in df.iloc[-1].iteritems():
        last_row_dict[attr_name] = attr_value

    if len(last_row_dict) == 0:
        return last_row_dict
    return last_row_dict


def __get_last_epoch_idx(saving_path):
    if saving_path is None or len(saving_path) == 0:
        full_path = "epochs_data.csv"
    else:
        full_path = f"{saving_path}/epochs_data.csv"

    last_epoch_idx = 0
    if os.path.exists(full_path):
        df = pd.read_csv(full_path, index_col=False)
        if df is not None:
            last_epoch_idx = len(df)
    return last_epoch_idx
