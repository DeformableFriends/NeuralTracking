from timeit import default_timer as timer


class TimeStatistics:
    def __init__(self):
        self.start_time = timer()
        self.io_duration = 0.0
        self.train_duration = 0.0
        self.eval_duration = 0.0
        self.forward_duration = 0.0
        self.loss_eval_duration = 0.0
        self.backward_duration = 0.0
    