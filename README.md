# 👁️ Blink Detection Backend (Flask)

This is the **Flask-based backend** for a real-time eye blink detection application. It uses **MediaPipe** for facial landmark detection and **OpenCV** for webcam access and image processing. The backend is built to work with any frontend (e.g., React) that can consume REST APIs and live video streams.

---

## 🧠 Features

- Real-time **blink detection** using Eye Aspect Ratio (EAR)
- Calculates **Blink Rate (BPM)** (blinks per minute)
- Filters out false positives due to **head movement**
- Displays a **sky blue box** around the face
- Provides **REST API endpoints** to manage detection state
- Optimized for **low-light conditions** and fast blinking
- Modular folder structure using `app.py`, `routes.py`, and `detection.py`

---

## 🛠 Tech Stack

- Python 3.x
- Flask
- OpenCV
- MediaPipe (FaceMesh)
- NumPy
- Flask-CORS

---

## 📂 Folder Structure
blink-detection-backend/
├── app.py # Flask app initialization and server entry point
├── routes.py # API route definitions
├── detection.py # Video stream and blink detection logic
├── requirements.txt # Python dependencies
└── README.md # Project description (this file)


---

## 🔌 REST API Endpoints

| Method | Endpoint         | Description                              |
|--------|------------------|------------------------------------------|
| GET    | `/`              | Loads homepage (optional for backend)    |
| GET    | `/video_feed`    | Returns MJPEG webcam stream              |
| GET    | `/stop`          | Stops the video detection                |
| GET    | `/blink_count`   | Returns total blink count                |
| GET    | `/reset`         | Resets all blink stats                   |
| GET    | `/face_status`   | Returns whether a face is detected       |
| GET    | `/health`        | Returns server health status             |

---

## 📦 Installation

Make sure Python 3 + is installed.




