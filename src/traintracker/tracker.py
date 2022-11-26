import os
import sys
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .hyperparameters import save_hpyerparameters
from .epochs_stats import EpochStatus
from torch.optim import Optimizer
from enum import Enum


class TrackerMod(Enum):
    TRAIN_ONLY = 0
    TEST_ONLY = 1
    TRAIN_TEST = 2


class TrainTracker:
    def __init__(self, model: torch.nn.Module, tracker_mod: TrackerMod = TrackerMod.TRAIN_TEST,
                 train_data_size: tuple = None, test_data_size: tuple = None,
                 train_data_dir: str = None,
                 hyperparameters: dict = None,
                 weights_dir: str = None):
        """
        :param model:pytorch Model
        :param tracker_mod: tracker mod instance from TrackerMod enum (TRAIN_ONLY,TEST_ONLY,TRAIN_TEST)
        :param train_data_size: tuple (n_train_batches, train_batch_size)
        :param test_data_size: tuple (n_test_batches, test_batch_size)
        :param hyperparameters: hyperparamters you which to track to be saved in excel_sheet train_data.csv
        :param train_data_dir:train data and epochs data csv file directory path
        :param weights_dir:model weights saving path
        """
        self.tracker_mod = tracker_mod
        self.model = model
        self.hyperparameters = hyperparameters
        self.weights_dir = weights_dir
        self.train_data_dir = train_data_dir
        self.current_process = 'test' if tracker_mod == TrackerMod.TEST_ONLY else 'train'
        if tracker_mod != TrackerMod.TEST_ONLY:
            # train and test or train only
            # add train data
            self.n_train_batches, self.train_batch_size = train_data_size
            self.train_epoch_status = EpochStatus(self.n_train_batches, self.train_batch_size)
            save_hpyerparameters(model, saving_path=train_data_dir, hyperparameters=hyperparameters)
            self.last_train_loss = 0
            self.train_data_size = self.n_train_batches * self.train_batch_size
            self.minTrainLoss = None

        if tracker_mod == TrackerMod.TEST_ONLY or tracker_mod == TrackerMod.TRAIN_TEST:
            # add test data
            self.n_test_batches, self.test_batch_size = test_data_size
            self.validation_epoch_status = EpochStatus(self.n_test_batches, self.test_batch_size)
            self.test_data_size = self.n_test_batches * self.test_batch_size
            self.minTestLoss = None

    def train(self) -> None:
        """
        change to tracking training loop mod to save the next steps in the trainEpoch status

        """
        self.current_process = "train"

    def valid(self):
        """
            change to tracking Validation loop mod to save the next steps in the ValidationEpoch status
        """
        self.current_process = "valid"

    def step(self, loss: float) -> float:
        """
        train or validation step according to current mod to add to the loss sum , calculate time remaining
        print current avg loss , remaining time , loading bar
        :param loss: current batch loss
        :return: avg loss
        """
        if self.current_process == "train":
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
        for train only
        - return epoch train loss
        - print epoch train loss  and time taken for train
        - save model weights if the train loss decreased

        :return:
        """

        if self.tracker_mod == TrackerMod.TRAIN_TEST:
            avg_test_loss, total_test_time = self.validation_epoch_status.last_epoch_summary()
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
                    save_train_weights(self.model, self.weights_dir, avg_train_loss, avg_test_loss)
                    print("achieved, model weights saved", end=" ")
                    print()
                    self.minTestLoss = avg_test_loss

            if avg_train_loss < avg_test_loss:
                print("!!!Warning Overfitting!!!")
            save_epoch_to_csv(self.train_data_dir, total_train_time + total_test_time, avg_train_loss,
                              self.train_data_size, avg_test_loss,
                              self.test_data_size)
            return avg_train_loss, avg_test_loss
        elif self.tracker_mod == TrackerMod.TRAIN_ONLY:
            avg_train_loss, total_train_time = self.train_epoch_status.last_epoch_summary()
            sys.stdout.flush()
            print(
                f" epoch {self.train_epoch_status.epoch_idx - 1} train_loss ={avg_train_loss}  total_time= {total_train_time}")
            if self.minTrainLoss is None:
                self.minTrainLoss = avg_train_loss
            else:
                if avg_train_loss < self.minTrainLoss:
                    print(
                        f"new minimum train loss {str(avg_train_loss)} ", end=" ")
                    save_train_weights(self.model, self.weights_dir, avg_train_loss)
                    print("achieved, model weights saved", end=" ")
                    print()
                    self.minTrainLoss = avg_train_loss

            save_epoch_to_csv(self.train_data_dir, total_train_time, avg_train_loss, self.train_data_size,
                              )
            return avg_train_loss

        else:
            avg_test_loss, total_test_time = self.validation_epoch_status.last_epoch_summary()
            print(
                f" avg_test_loss ={avg_test_loss} total_time= {total_test_time}")
            return avg_test_loss


def save_train_weights(model: torch.nn.Module, saving_path: str, train_loss: float = None, test_loss: float = None):
    """
    saves model weights with file name format Day_Month Hour_minute train_(train_loss) test_(test_loss)
    :param model: model object
    :param saving_path: the path you want to save the weights in
    :param train_loss: train loss (float)
    :param test_loss: test loss (float)

    :return: the full path of the saved file (saving_path+filename)
    """
    weight_file_name = f"{datetime.now().strftime('%m_%d %H_%M')} "
    if train_loss is not None:
        weight_file_name += f"Train_({round(train_loss, 5)}) "
    if test_loss is not None:
        weight_file_name += f"Test_({round(test_loss, 5)})"
    weight_file_name += ".pt"
    full_path = f"{saving_path}/{weight_file_name}"

    torch.save(model.state_dict(), full_path)
    return full_path


def save_epoch_to_csv(saving_path: str, time_taken: float, train_loss: float, no_train_rows: int,
                      test_loss: float = None,
                      no_test_rows: int = None,
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

    row = [train_loss, no_train_rows]
    cols = ["Train Loss", "no train rows"]
    if test_loss is not None:
        cols += ["Test Loss", "No test rows"]
        row += [test_loss, no_test_rows]
    cols += ["Time taken (M)", "Date", "Time"]
    row += [time_taken, date_now.strftime('%d/%m/%Y'), date_now.strftime('%H:%M:00')]
    df = pd.DataFrame([row],
                      columns=cols)

    if not os.path.exists(full_path):
        df.to_csv(full_path, index=False)
    else:
        df.to_csv(full_path, mode='a', header=False, index=False)
