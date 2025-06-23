👁️ Blink Detection Backend (Flask Only)
This is the Flask-based backend for a real-time eye blink detection system using MediaPipe and OpenCV. It provides REST APIs that accept image frames and return blink count and face detection results. Supports multiple clients concurrently via client_id with persistent state storage.

🧠 Features
Real-time blink detection using Eye Aspect Ratio (EAR)
Uses MediaPipe FaceMesh for landmark detection
Detects and filters out head movements
Supports multiple users via client_id
Simple, modular Flask structure

🛠 Tech Stack
Python 3.10.11
Flask
Flask-CORS
OpenCV
MediaPipe
NumPy
SciPy

📁 Folder Structure
blink-detection-backend/
├── app.py             # Entry point of the Flask server
├── routes.py          # API routes for detection and reset
├── detection.py       # Core blink detection logic
├── requirements.txt   # Python dependencies
└── README.md          # Documentation (this file)


📦 Installation
Clone the repository
git clone <your-repo-url>
cd blink-detection-backend

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt
Or manually:
pip install flask flask-cors opencv-python mediapipe scipy numpy

▶️ Run the Server
python app.py
Server will start at: http://127.0.0.1:5000

🔌 API Endpoints
POST /upload_frame
Send a single base64-encoded frame for blink detection.

Request Body:
{
  "client_id": "your_unique_id",
  "image": "data:image/png;base64,..."  // Full base64 image string
}

Response:
{
  "blink_count": 2,
  "face_detected": true
}


POST /reset
Reset the blink count and detection state for a specific client.

Request Body:
{ "client_id": "your_unique_id" }

Response:
{ "status": "reset" }

GET /health
Health check endpoint.

Response:
{ "status": "ok" }


✅ requirements.txt
flask
flask-cors
opencv-python
mediapipe
scipy
numpy
