# 🚨 CrimeVision-Net: Real-Time CCTV Crime Detection using VideoMAE

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-orange.svg)
![License](https://img.shields.io/badge/License-MIT-blue)

**Real-Time Intelligent CCTV Crime Detection and Alert System**

</div>

---

# 📌 Overview

CrimeVision-Net is an AI-powered surveillance system that automatically detects suspicious and criminal activities from CCTV video streams.

The system combines:

- YOLO11 object detection
- ByteTrack multi-object tracking
- VideoMAE action recognition
- Real-time alert stabilization

to identify criminal activities such as:

- Assault
- Fighting
- Robbery
- Shoplifting
- Road Accident
- Burglary
- Arson
- Shooting
- Vandalism
- Stealing
- Explosion
- Arrest
- Abuse
- Normal Activity

Unlike traditional surveillance, CrimeVision-Net continuously monitors live video streams and generates alerts only after multiple consistent predictions, reducing false alarms.

---

# 🎯 Features

- Real-time CCTV monitoring
- Person and vehicle detection
- Multi-object tracking
- Temporal action recognition using VideoMAE
- Alert stabilization
- Live visualization with bounding boxes
- RTSP Camera Support
- Webcam Support
- Video File Support
- GPU Acceleration
- Easy model replacement

---

# 🏗 System Architecture

```

Live CCTV Stream
│
▼
YOLO11 Detector
│
▼
ByteTrack Tracker
│
▼
Track Buffer
│
▼
VideoMAE Classifier
│
▼
Alert Stabilizer
│
▼
Monitoring Dashboard

```

---

# 🧠 Model

## Detection

- YOLO11 (Ultralytics)

## Tracking

- ByteTrack

## Action Recognition

- VideoMAE Base
- Fine-tuned on the UCF-Crime Dataset

---

# 📂 Dataset

The model is trained using the **UCF-Crime Dataset**.

Dataset Classes:

- Abuse
- Arrest
- Arson
- Assault
- Burglary
- Explosion
- Fighting
- Road Accident
- Robbery
- Shooting
- Shoplifting
- Stealing
- Vandalism
- Normal

---

# 📊 Performance

| Metric | Value |
|---------|-------|
| Backbone | VideoMAE Base |
| Dataset | UCF-Crime |
| Number of Classes | 14 |
| Validation Accuracy | **69.07%** |

---

# 📁 Project Structure

```
CrimeVision-Net
│
├── checkpoints/
├── data/
├── detector.py
├── classifier.py
├── dataset.py
├── prepare_data.py
├── pipeline.py
├── track_buffer.py
├── alert_logic.py
├── train.py
├── config.py
├── requirements.txt
└── README.md
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/CrimeVision-Net.git

cd CrimeVision-Net
```

Create virtual environment

```bash
python -m venv .venv
```

Activate environment

Windows

```bash
.venv\Scripts\activate
```

Linux

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Training

Prepare dataset

```bash
python prepare_data.py --root data
```

Train VideoMAE

```bash
python train.py \
--manifest data/manifest.csv \
--epochs 10 \
--batch_size 4
```

---

# 🎥 Real-Time Inference

Using Webcam

```bash
python pipeline.py --source 0
```

Using Video

```bash
python pipeline.py --source video.mp4
```

Using RTSP Camera

```bash
python pipeline.py --source rtsp://username:password@ip/live
```

---

# 🔄 Pipeline

1. Capture frame
2. Detect people and vehicles
3. Track objects
4. Build temporal clip
5. Classify action
6. Stabilize prediction
7. Generate alert
8. Display annotated video

---

# 💡 Future Improvements

- Mobile notification system
- Security dashboard
- Cloud deployment
- Multi-camera support
- Face recognition integration
- Weapon detection
- Person re-identification
- Official UCF-Crime train/test split
- Lightweight real-time backbone (X3D, MoViNet)

---

# 🛠 Technologies

- Python
- PyTorch
- HuggingFace Transformers
- VideoMAE
- YOLO11
- ByteTrack
- OpenCV
- NumPy
- Pandas

---

# 📈 Results

The proposed CrimeVision-Net framework successfully learns temporal representations of criminal activities from surveillance videos.

The fine-tuned VideoMAE model achieved:

> **Validation Accuracy: 69.07%**

while maintaining compatibility with a real-time surveillance pipeline consisting of object detection, tracking, temporal action recognition, and alert stabilization.

---

# 📚 Citation

If you use this project in your research, please cite:

```
@software{CrimeVisionNet,
  title={CrimeVision-Net: Real-Time CCTV Crime Detection using VideoMAE},
  author={Mutte Ullah},
  year={2026}
}
```

---

# 👨‍💻 Author

**Mutte Ullah**

BS Artificial Intelligence

Islamia University Bahawalpur

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.
