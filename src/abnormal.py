import numpy as np

class AbnormalMotion:
    def __init__(self):
        self.prev = {}

    def check(self, pid, point):
        if pid not in self.prev:
            self.prev[pid] = point
            return False, 0

        prev = self.prev[pid]
        vel = np.linalg.norm(np.array(point) - np.array(prev))
        self.prev[pid] = point

        return vel > 60, vel
