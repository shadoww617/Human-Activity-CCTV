# 🛡️ Human Activity Recognition CCTV System
A smart AI-powered surveillance system that analyzes CCTV footage to detect:

- ✔️ People & Pose Detection (YOLOv8 Pose)
- ✔️ Loitering Detection
- ✔️ Abnormal Motion Detection
- ✔️ Violence / Aggressive Activity Indicators
- ✔️ Movement Trajectory Visualization
- ✔️ Scene Understanding with Reasoning
- ✔️ Real-time Streamlit Dashboard

This project combines **Computer Vision + Machine Learning + Data Analytics** to simulate a real-world intelligent CCTV monitoring system.

---

## 🚀 Features
### 🎥 Real-Time Surveillance Intelligence
- Detects humans using YOLO
- Tracks individuals and paths
- Identifies behavior patterns
- Generates alerts intelligently

### 🧠 AI Reasoning Engine
The system explains decisions:
- Tracks risk history
- Builds confidence
- Provides readable explanations
- Scene inference (Normal / Suspicious / High Risk)

### 📊 Visual Analytics
- Live video feed with detections
- Movement trajectory map
- Alerts panel
- Behavior inference panel

---

## 🛠️ Tech Stack
| Component | Tech |
|---------|-------|
| Detection | YOLOv8 |
| Pose Estimation | YOLO Pose |
| ML Reasoning | Python + Numpy |
| Dashboard | Streamlit |
| GPU Acceleration | PyTorch CUDA |
| Tracking | Custom Tracker + Trajectory Engine |

---

## 📂 Project Structure
Human-Activity-CCTV/
│
├── src/
│ ├── detector_yolo.py
│ ├── pose_extractor.py
│ ├── tracker.py
│ ├── loitering.py
│ ├── abnormal.py
│ ├── trajectory_map.py
│ ├── trajectory_inference.py
│ ├── violence_inference.py
│
├── streamlit_app.py
├── requirements.txt
├── README.md

---

##⚡ Install Dependencies
pip install -r requirements.txt

If GPU available:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Run Application
streamlit run streamlit_app.py

Open in browser:
http://localhost:8501

---

##🎯 Usage

Upload either:
- CCTV Video
- CCTV Image

Dashboard shows:
- Detections
- Intelligence reasoning
- Trajectories
- Alerts
- Scene understanding

---

##🧪 Testing Video Sources

Try:
- RWF-2000 violence dataset
- UCSD Pedestrian dataset
- UCF Crime dataset
- CCTV sample YouTube clips

---

##📌 Future Enhancements

- DeepSORT tracking
- Real violence CNN classifier
- Zone violation detection
- Report export (PDF/CSV)
- Save processed video