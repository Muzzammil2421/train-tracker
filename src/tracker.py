import os
import sys
from datetime import datetime
import pandas as pd
import torch

from .hyperparameters import save_hpyerparameters
from epochs_stats import EpochStatus


class TrainTracker:
    def __init__(self, model, train_loader, validation_loader, loss_function, optimizer, train_data_dir, weights_dir,
                 extra_info=None, notes=None):
        if notes is not None:
            extra_info["notes"] = notes

        self.model, self.train_loader, self.validation_loader, self.loss_function, self.optimizer, \
        self.train_data_dir, self.weights_dir = model, train_loader, validation_loader, loss_function, optimizer, train_data_dir, weights_dir
        save_hpyerparameters(model, saving_path=train_data_dir, batch_size=train_loader.batch_size,
                             optimizer=optimizer, loss_function=loss_function, extra_info=extra_info)

        self.train_epoch_status = EpochStatus(train_loader)
        self.validation_epoch_status = EpochStatus(validation_loader)
        self.bar_length = 10

        self.loading_bar = "[" + (self.bar_length * ".") + "]"
        self.mod = "train"
        self.minTestLoss = None
        self.last_train_loss = 0
        self.last_test_loss = 0

    def train(self):
        self.mod = "train"

    def valid(self):
        self.mod = "valid"

    def step(self, loss):
        if self.mod == "train":
            avg_train_loss, time_remaining = self.train_epoch_status.step(loss)

            sys.stdout.write("\r epoch " + str(
                self.train_epoch_status.epoch_idx) + self.train_epoch_status.get_loading_bar() + "time remaining (m) = " + str(
                time_remaining) + " Avg Train_Loss=" + str(avg_train_loss))

            if self.train_epoch_status.epoch_finished():
                # last forward step in the epoch print summary and new line
                avg_loss, total_time = self.train_epoch_status.epoch_summary()
                sys.stdout.flush()
                sys.stdout.write("\r epoch " + str(self.train_epoch_status.epoch_idx) + "] time Taken (git m) = " + str(
                    total_time) + " Avg Train_Loss=" + str(avg_loss))
                print()
        else:
            avg_test_loss, time_remaining = self.validation_epoch_status.step(loss)

            sys.stdout.write(
                "\r testing " + self.validation_epoch_status.get_loading_bar() + "time remaining (m) = " + str(
                    time_remaining) + " Avg Test_Loss=" + str(avg_test_loss))

            if self.validation_epoch_status.epoch_finished():
                # last forward step in the epoch print summary and new line

                avg_loss, total_time = self.validation_epoch_status.epoch_summary()
                sys.stdout.flush()
                sys.stdout.write(
                    "\r Testing " + str(self.validation_epoch_status.epoch_idx) + "] time Taken (git m) = " + str(
                        total_time) + " Avg Test_Loss=" + str(avg_loss))
                print()

    def end_epoch(self):
        # show test and train loss

        avg_train_loss, total_train_time = self.train_epoch_status.epoch_summary()

        avg_test_loss, total_test_time = self.validation_epoch_status.epoch_summary()

        if self.minTestLoss is None:
            self.minTestLoss = avg_test_loss
        else:
            if avg_test_loss < self.minTestLoss:
                print(
                    f"new minimum test loss {str(avg_test_loss)} ", end=" ")
                save_train_weights(self.model, avg_train_loss, avg_test_loss, self.weights_dir)
                print("achieved, model weights saved", end=" ")
                print()
                self.minTestLoss = avg_test_loss
        print(
            f" epoch {self.train_epoch_status.epoch_idx} train_loss ={avg_train_loss} test_loss={avg_test_loss} total_time= {total_train_time + total_test_time}")
        if avg_train_loss < avg_test_loss:
            print("!!!Warning Overfitting!!!")
        save_epoch_to_csv(self.train_data_dir, avg_train_loss, len(self.train_loader.dataset), avg_test_loss,
                          len(self.validation_loader.dataset), total_train_time + total_test_time)


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
