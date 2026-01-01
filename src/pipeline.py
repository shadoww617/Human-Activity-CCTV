import cv2
from detector_yolo import PersonDetector
from pose_extractor import extract_pose
from tracker import Tracker
from loitering import LoiterMonitor
from abnormal import AbnormalMotion
from heatmap import update_heatmap

detector = PersonDetector()
tracker = Tracker()
loiter = LoiterMonitor()
abnormal = AbnormalMotion()

def run(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        people = detector.detect(frame)
        tracked = tracker.update(people)

        for pid,(cx,cy,box) in tracked.items():
            _ = extract_pose(frame)

            loiter_flag = loiter.check(pid,(cx,cy))
            abnormal_flag = abnormal.check(pid,(cx,cy))

            update_heatmap(cx,cy)

            label = "normal"
            color = (0,255,0)

            if loiter_flag:
                label = "loitering"
                color = (0,200,255)

            if abnormal_flag:
                label = "abnormal"
                color = (0,0,255)

            cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),color,2)
            cv2.putText(frame,label,(box[0],box[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        cv2.imshow("CCTV Surveillance",frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
