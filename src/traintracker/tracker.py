import os
import sys
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .hyperparameters import save_hpyerparameters
from .epochs_stats import EpochStatus
from torch.optim import Optimizer


class TrainTracker:

    def __init__(self, model: torch.nn.Module, validation_loader: DataLoader, train_loader: DataLoader = None,
                 loss_function=None, optimizer: Optimizer = None,
                 train_data_dir: str = None, weights_dir: str = None,
                 extra_info: dict = None, notes: str = None):
        """
        :param model:pytorch Model
        :param validation_loader: Validation Data loader
        :param train_loader:Training Data Loader
        :param loss_function: Loss function used in the training
        :param optimizer:
        :param train_data_dir:train data and epochs data csv file directory path
        :param weights_dir:model weights saving path
        :param extra_info: extra info to be tracked and added in the train data csv file to monitor it's change
        :param notes: notes you which to add on the training data for the next epochs
        """

        if train_loader is None:
            # test mode only
            self.model, validation_loader = model, validation_loader
            self.test_only = True
            self.validation_epoch_status = EpochStatus(validation_loader)
            self.mod = "test"

        else:
            if notes is not None:
                extra_info["notes"] = notes

            self.model, self.train_loader, self.validation_loader, self.loss_function, self.optimizer, \
            self.train_data_dir, self.weights_dir = model, train_loader, validation_loader, loss_function, optimizer, train_data_dir, weights_dir
            save_hpyerparameters(model, saving_path=train_data_dir, batch_size=train_loader.batch_size,
                                 optimizer=optimizer, loss_function=loss_function, extra_info=extra_info)

            self.train_epoch_status = EpochStatus(train_loader)
            self.validation_epoch_status = EpochStatus(validation_loader)
            self.mod = "train"
            self.minTestLoss = None
            self.last_train_loss = 0
            self.last_test_loss = 0
            self.test_only = False

    def train(self) -> None:
        """
        change to tracking training loop mod to save the next steps in the trainEpoch status

        """
        self.mod = "train"

    def valid(self):
        """
            change to tracking Validation loop mod to save the next steps in the ValidationEpoch status
        """
        self.mod = "valid"

    def step(self, loss: float) -> float:
        """
        train or validation step according to current mod to add to the loss sum , calculate time remaining
        print current avg loss , remaining time , loading bar
        :param loss: current batch loss
        :return: avg loss
        """
        if self.mod == "train":
            avg_loss, time_remaining = self.train_epoch_status.step(loss)

            if self.train_epoch_status.epoch_finished():
                # last forward step in the epoch print summary and new line

                avg_loss, total_time = self.train_epoch_status.last_epoch_summary()

                sys.stdout.write(
                    "\r training epoch " + str(self.train_epoch_status.epoch_idx) + " time Taken (m) = " + str(
                        total_time) + " Avg Train_Loss=" + str(avg_loss))
                print()
                sys.stdout.flush()
                self.train_epoch_status.next_epoch()

            else:
                sys.stdout.write("\r epoch " + str(
                    self.train_epoch_status.epoch_idx) + self.train_epoch_status.get_loading_bar() + "time remaining (m) = " + str(
                    time_remaining) + " Avg Train_Loss=" + str(avg_loss))
                sys.stdout.flush()


        else:
            avg_loss, time_remaining = self.validation_epoch_status.step(loss)

            if self.validation_epoch_status.epoch_finished():
                # last forward step in the epoch print summary and new line

                avg_loss, total_time = self.validation_epoch_status.last_epoch_summary()
                sys.stdout.write(
                    "\r Test  time Taken (m) = " + str(
                        total_time) + " Avg Test_Loss=" + str(avg_loss))
                print()
                sys.stdout.flush()
                self.validation_epoch_status.next_epoch()

            else:
                sys.stdout.write(
                    "\r testing " + self.validation_epoch_status.get_loading_bar() + " time remaining (m) = " + str(
                        time_remaining) + " Avg Test_Loss=" + str(avg_loss))
                sys.stdout.flush()

        return avg_loss

    def end_epoch(self):
        """
         for train and validation
        - get epoch last epoch train loss and validation loss
        - print epoch train and validation loss and time taken for train and validation
        - save model weights if the test loss decreased
        for test only
        - - get epoch test loss

        :return:
        """

        avg_test_loss, total_test_time = self.validation_epoch_status.last_epoch_summary()

        if not self.test_only:
            avg_train_loss, total_train_time = self.train_epoch_status.last_epoch_summary()
            sys.stdout.flush()
            print(
                f" epoch {self.train_epoch_status.epoch_idx - 1} train_loss ={avg_train_loss} test_loss={avg_test_loss} total_time= {total_train_time + total_test_time}")
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

            if avg_train_loss < avg_test_loss:
                print("!!!Warning Overfitting!!!")
            save_epoch_to_csv(self.train_data_dir, avg_train_loss, len(self.train_loader.dataset), avg_test_loss,
                              len(self.validation_loader.dataset), total_train_time + total_test_time)
            return avg_train_loss, avg_test_loss
        else:
            print(
                f" avg_test_loss ={avg_test_loss} total_time= {total_test_time}")
            return avg_test_loss


def save_train_weights(model: torch.nn.Module, train_loss: float, test_loss: float, saving_path: str):
    """
    saves model weights with file name format Day_Month Hour_minute train_(train_loss) test_(test_loss)
    :param model: model object
    :param train_loss: train loss (float)
    :param test_loss: test loss (float)
    :param saving_path: the path you want to save the weights in
    :return: the full path of the saved file (saving_path+filename)
    """
    weight_file_name = f"{datetime.now().strftime('%m_%d %H_%M')} Train_({round(train_loss, 5)}) Test_({round(test_loss, 5)}).pt"
    full_path = f"{saving_path}/{weight_file_name}"

    torch.save(model.state_dict(), full_path)
    return full_path


def save_epoch_to_csv(saving_path: str, train_loss: float, no_train_rows: int, test_loss: float, no_test_rows: int,
                      time_taken: float,
                      ) -> None:
    """
    append current epoch data to existing csv file if the file doesn't exist create new one
    :param saving_path:csv folder path
    :param train_loss: epoch train loss
    :param no_train_rows: no of train rows (not batches)
    :param test_loss: epoch test loss
    :param no_test_rows: no of test rows (not batches)
    :param time_taken: time taken for training and testing for this epoch
    """
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
