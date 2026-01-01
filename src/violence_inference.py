import time
import numpy as np

class ViolenceInference:
    def __init__(self):
        self.motion_history = {}
        self.last_flag_time = 0
        self.cooldown = 2  # seconds

    def update(self, pid, velocity):
        t = time.time()

        if pid not in self.motion_history:
            self.motion_history[pid] = []

        self.motion_history[pid].append(velocity)

        if len(self.motion_history[pid]) > 20:
            self.motion_history[pid].pop(0)

        # compute aggression score
        arr = np.array(self.motion_history[pid])

        mean = np.mean(arr)
        spikes = np.sum(arr > 50)

        score = mean + (spikes * 8)

        if score > 70 and (t - self.last_flag_time) > self.cooldown:
            self.last_flag_time = t
            return True

        return False
