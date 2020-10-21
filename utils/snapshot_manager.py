import os
import struct
import json
import numpy as np
import torch
from timeit import default_timer as timer
from datetime import datetime


class SnapshotManager():
    def __init__(self, log_name, model_dir, snapshot_duration=5):
        self.snapshot_duration = snapshot_duration
        self.log_name = log_name
        self.model_dir = model_dir
        self.starting_time = timer()
        self.previous_snapshot_time = timer()

    def save_model(self, model, iteration_number, final_iteration=False):
        time_current = timer()
        elapsed = time_current - self.previous_snapshot_time

        print()

        if final_iteration:
            elapsed = time_current - self.starting_time
            if elapsed < self.snapshot_duration:
                print("Not enough time elapsed ({}s < {}s) to store a new snapshot.".format(elapsed, self.snapshot_duration))
                return
        elif elapsed < self.snapshot_duration:
            print("Not enough time elapsed ({}s < {}s) to store a new snapshot.".format(elapsed, self.snapshot_duration))
            return

        # Enough time elapsed to store a new snapshot.
        model_name = "{}_{}.pt".format(self.log_name, iteration_number)
        model_path = os.path.join(self.model_dir, model_name)
        torch.save(model.state_dict(), model_path)

        print("Enough time elapsed ({}s >= {}s) to store a new snapshot.".format(elapsed, self.snapshot_duration))
        print("Saving model as {}".format(model_name))
        print()

        self.previous_snapshot_time = timer()
    

if __name__ == "__main__":
    pass