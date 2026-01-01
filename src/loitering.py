import time

class LoiterMonitor:
    def __init__(self):
        self.history = {}

    def check(self, pid, point):
        if pid not in self.history:
            self.history[pid] = {"pos": point, "time": time.time()}
            return False
        
        old = self.history[pid]

        dist = abs(point[0] - old["pos"][0]) + abs(point[1] - old["pos"][1])

        if dist < 30:
            if time.time() - old["time"] > 10:
                return True
        else:
            self.history[pid] = {"pos": point, "time": time.time()}

        return False
