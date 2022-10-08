from time import time
from torch.utils.data import DataLoader


class EpochStatus:
    bar_length = 10

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.epoch_idx = 1

        self.forward_cnt = 0
        self.loss_sum = 0.0
        self.time_sum = 0.0
        self.start_time = time()
        self.last_step_time = time()

        self.last_epoch_total_time = 0.0
        self.last_epoch_avg_loss = 0.0
        # TODO: separate start time with start loop function and reset data at the start of the loop

    def step(self, loss: float):
        """
        get batch loss sum , add it to the epoch loss sum , calculate avg loss
        calculate time for forward step , add it to total time taken , calc avg time per forward step

        :param loss:
        :return:
        """
        self.loss_sum += (loss * self.data_loader.batch_size)
        self.forward_cnt += 1.0

        forward_fin_time = time()

        step_time = forward_fin_time - self.last_step_time
        self.time_sum += step_time
        self.last_step_time = forward_fin_time

        avg_step_time = self.time_sum / self.forward_cnt
        time_remaining = avg_step_time * (len(self.data_loader) - self.forward_cnt)
        avg_loss = round(self.loss_sum / (self.forward_cnt * self.data_loader.batch_size), 8)
        if self.epoch_finished():
            self.last_epoch_avg_loss = avg_loss
            self.last_epoch_total_time = time() - self.start_time

        return avg_loss, self.sec2min(time_remaining)

    def epoch_finished(self) -> bool:
        return self.forward_cnt >= len(self.data_loader)

    def next_epoch(self):
        """
        reset epoch status data (loss_sum,total_time)
        :return:
        """
        self.forward_cnt = 0
        self.loss_sum = 0.0
        self.time_sum = 0.0
        self.start_time = time()
        self.last_step_time = time()
        self.epoch_idx += 1

    def get_loading_bar(self) -> str:
        """
        creates a loading string bar [=====...........]
        :return:
        """
        finished_procedure = int((self.forward_cnt * self.bar_length) / len(self.data_loader))
        remaining_procedure = self.bar_length - finished_procedure
        return "[" + ("=" * finished_procedure) + (remaining_procedure * ".") + "]"

    def last_epoch_summary(self):
        """
        get the past epoch avg_loss and time taken
        :return: avg loss (float),avg time taken  (float)
        """
        return self.last_epoch_avg_loss, self.sec2min(self.last_epoch_total_time)

    @staticmethod
    def sec2min(seconds: float):
        return round(seconds / 60.0, 2)
