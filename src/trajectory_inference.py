import numpy as np

class TrajectoryReasoner:
    def __init__(self):
        self.tracks = {}

    def update(self, pid, point):
        if pid not in self.tracks:
            self.tracks[pid] = []

        self.tracks[pid].append(point)

        if len(self.tracks[pid]) > 40:
            self.tracks[pid].pop(0)

        track = np.array(self.tracks[pid])

        if len(track) < 5:
            return "Analyzing..."

        # movement magnitude
        movement = np.linalg.norm(track[-1] - track[0])

        # direction variance
        diffs = np.diff(track, axis=0)
        directions = np.arctan2(diffs[:,1], diffs[:,0])
        dir_var = np.var(directions)

        # inference rules
        if movement < 20:
            return "Standing / Possible Loitering"

        if dir_var < 0.1:
            return "Normal Walking"

        if dir_var > 1.5:
            return "Chaotic / Suspicious Movement"

        if movement > 150 and dir_var > 0.5:
            return "Fast Escape Motion"

        return "Normal Behavior"
