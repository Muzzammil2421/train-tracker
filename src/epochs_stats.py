from time import time


class EpochStatus:
    def __init__(self, data_loader):
        self.bar_length = 10
        self.data_loader = data_loader
        self.epoch_idx = 1

        self.forward_cnt = 0
        self.loss_sum = 0.0
        self.time_sum = 0.0
        self.start_time = time()
        self.last_step_time = time()
        # [*****************************]#
        self.loading_bar = "[" + (self.bar_length * ".") + "]"

    def step(self, loss):
        self.loss_sum += loss
        self.forward_cnt += 1.0

        forward_fin_time = time()

        step_time = forward_fin_time - self.last_step_time
        self.time_sum += step_time
        self.last_step_time = forward_fin_time

        avg_step_time = self.time_sum / self.forward_cnt
        time_remaining = avg_step_time * (len(self.data_loader) - self.forward_cnt)
        avg_loss = round(self.loss_sum / (self.forward_cnt * self.data_loader.batch_size), 8)
        if self.forward_cnt==len(self.data_loader):
            self.epoch_idx += 1
        return avg_loss, self.sec2min(time_remaining)

    def epoch_finished(self) -> bool:
        return self.forward_cnt == len(self.data_loader)

    def reset(self):
        self.forward_cnt = 0
        self.loss_sum = 0.0
        self.time_sum = 0.0
        self.start_time = time()
        self.last_step_time = time()

    def get_loading_bar(self) -> str:
        finished_procedure = int((self.forward_cnt * self.bar_length) / len(self.data_loader))
        remaining_procedure = self.bar_length - finished_procedure
        return "[" + ("=" * finished_procedure) + (remaining_procedure * ".") + "]"

    def epoch_summary(self):
        total_time = time() - self.start_time
        avg_loss = round(self.loss_sum / (self.forward_cnt * self.data_loader.batch_size), 8)
        return avg_loss, total_time

    @staticmethod
    def sec2min(seconds: float):
        return round(seconds / 60.0, 2)
