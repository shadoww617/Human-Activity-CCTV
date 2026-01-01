import streamlit as st
import cv2
import numpy as np

from src.detector_yolo import PersonDetector
from src.pose_extractor import extract_pose
from src.tracker import Tracker
from src.loitering import LoiterMonitor
from src.abnormal import AbnormalMotion

from src.trajectory_map import ensure_canvas, update_trajectory, get_canvas
from src.trajectory_inference import TrajectoryReasoner
from src.violence_inference import ViolenceInference

st.set_page_config(page_title="Human Activity CCTV", layout="wide")

st.title("🛡️ Human Activity Recognition CCTV System")
st.markdown("#### Detects Normal | Loitering | Abnormal Motion | Violence | Trajectory Intelligence")

# ===== Session Memory =====
if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "reason_log" not in st.session_state:
    st.session_state.reason_log = []

if "risk_score" not in st.session_state:
    st.session_state.risk_score = 0


def reset_state():
    st.session_state.alerts = []
    st.session_state.reason_log = []
    st.session_state.risk_score = 0


col1, col2 = st.columns(2)

with col1:
    video_file = st.file_uploader("🎥 Upload Video",
                                  type=["mp4", "avi", "mov"],
                                  on_change=reset_state)

with col2:
    image_file = st.file_uploader("🖼️ Upload Image",
                                  type=["jpg", "jpeg", "png"],
                                  on_change=reset_state)

detector = PersonDetector()
tracker = Tracker()
loiter = LoiterMonitor()
abnormal = AbnormalMotion()
traj_reason = TrajectoryReasoner()
violence_brain = ViolenceInference()

video_placeholder = st.empty()
alert_box = st.empty()
trajectory_placeholder = st.empty()
inference_box = st.empty()
history_box = st.empty()


def show_trajectory_panel():
    traj = cv2.cvtColor(get_canvas(), cv2.COLOR_BGR2RGB)
    trajectory_placeholder.image(traj, caption="🧭 Movement Trajectory Map")


# ===== Image Mode =====
if image_file:
    st.subheader("🖼️ Image Analysis Mode")
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    people = detector.detect(frame)
    tracked = tracker.update(people)

    for pid, (cx, cy, box) in tracked.items():
        _ = extract_pose(frame)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                      (0, 255, 0), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption="Processed Image", use_column_width=True)


# ===== Video Mode =====
if video_file:
    st.subheader("🎥 Video Analysis Mode")
    temp_video = "input.mp4"

    with open(temp_video, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(temp_video)
    ret, frame = cap.read()

    ensure_canvas(frame.shape[0], frame.shape[1])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    st.success("🚀 Processing on GPU")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        people = detector.detect(frame)
        tracked = tracker.update(people)

        violent_count = 0
        frame_reason_notes = []

        for pid, (cx, cy, box) in tracked.items():

            _ = extract_pose(frame)

            loiter_flag = loiter.check(pid, (cx, cy))
            abnormal_flag, vel = abnormal.check(pid, (cx, cy))

            update_trajectory(pid, (cx, cy))
            behavior = traj_reason.update(pid, np.array([cx, cy]))

            violence_flag = violence_brain.update(pid, vel)

            label = behavior
            color = (0, 255, 0)

            if loiter_flag:
                label = "Loitering Suspicion"
                color = (0, 200, 255)
                frame_reason_notes.append("Person remained stationary unusually long.")
                st.session_state.risk_score += 2

            if abnormal_flag:
                label = "Abnormal Motion"
                color = (0, 0, 255)
                frame_reason_notes.append("Velocity spike: sudden abnormal movement detected.")
                st.session_state.risk_score += 3

            if violence_flag:
                violent_count += 1
                label = "⚠️ Possible Fight / Aggressive Motion"
                color = (255, 0, 0)
                frame_reason_notes.append("Chaotic aggressive movement pattern detected.")
                st.session_state.risk_score += 5

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ===== Aggregate Scene Conclusion =====
        if frame_reason_notes:
            st.session_state.reason_log.extend(frame_reason_notes)

        if violent_count >= 2:
            scene_status = "🚨 High Risk: Possible Fight"
        elif st.session_state.risk_score > 20:
            scene_status = "⚠️ Suspicious Activity Persistent"
        elif st.session_state.risk_score > 10:
            scene_status = "⚠️ Multiple Risk Events Observed"
        else:
            scene_status = "✔️ Mostly Normal Scene"

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", width=700)

        for alert in st.session_state.alerts[-6:]:
            alert_box.error(alert)

        show_trajectory_panel()

        # ===== Display Final Reasoning Panel =====
        explanation = "\n".join(st.session_state.reason_log[-8:]) if st.session_state.reason_log else "No threatening patterns consistently detected."

        inference_box.info(
            f"**Scene Understanding:** {scene_status}\n\n"
            f"**Cumulative Reasoning History:**\n{explanation}\n\n"
            f"**Risk Score:** {st.session_state.risk_score}"
        )

    cap.release()
