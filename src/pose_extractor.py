from ultralytics import YOLO

pose_model = YOLO("yolov8n-pose.pt")

def extract_pose(frame):
    results = pose_model.predict(
        frame,
        conf=0.4,
        verbose=False,
        device="cuda",
        half=True
    )

    keypoints = []

    for r in results:
        if r.keypoints is None:
            continue
        
        for kp in r.keypoints:
            pts = kp.xy[0].tolist()
            flat = []
            for p in pts:
                flat.extend([p[0], p[1]])
            keypoints.append(flat)

    return keypoints if keypoints else None
