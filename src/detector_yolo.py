from ultralytics import YOLO

class PersonDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):
        results = self.model.predict(
            frame,
            conf=0.4,
            verbose=False,
            device="cuda",
            half=True
        )

        people = []

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:   # person
                    x1, y1, x2, y2 = box.xyxy[0]
                    people.append((int(x1), int(y1), int(x2), int(y2)))

        return people
