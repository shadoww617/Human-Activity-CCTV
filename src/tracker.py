class Tracker:
    def __init__(self):
        self.people = {}

    def update(self, boxes):
        new_data = {}
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            new_data[i] = (cx, cy, box)

        self.people = new_data
        return self.people
