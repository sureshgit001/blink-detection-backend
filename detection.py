import cv2, time
from scipy.spatial import distance
import mediapipe as mp
import numpy as np

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Constants
EYE_AR_THRESH = 0.21
MIN_BLINK_FRAMES = 2
RECOVERY_FRAMES = 3
MOVEMENT_THRESH = 15

# Globals
blink_count = 0
counter = 0
blink_detected = False
frames_recovery = 0
blink_times = []
prev_landmarks = None
face_detected = False
detection_active = False
cap = None

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C else 0.0

def detect_movement(curr_landmarks, prev_landmarks):
    coords = [(lm.x, lm.y) for lm in curr_landmarks]
    prev_coords = [(lm.x, lm.y) for lm in prev_landmarks]
    diffs = [abs(c[0]-p[0]) + abs(c[1]-p[1]) for c, p in zip(coords, prev_coords)]
    return np.mean(diffs) * 1000

def generate():
    global blink_count, counter, blink_detected, frames_recovery, blink_times
    global prev_landmarks, face_detected, detection_active, cap

    cap = cv2.VideoCapture(0)
    try:
        while detection_active:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            face_detected = bool(results.multi_face_landmarks)

            if face_detected:
                lm_list = results.multi_face_landmarks[0].landmark

                # 🟦 Draw face box
                x_coords = [int(lm.x * w) for lm in lm_list]
                y_coords = [int(lm.y * h) for lm in lm_list]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (235, 206, 135), 2)

                mv = detect_movement(lm_list, prev_landmarks) if prev_landmarks else 0
                prev_landmarks = list(lm_list)

                if mv > MOVEMENT_THRESH:
                    counter = 0
                    blink_detected = False
                    frames_recovery = 0
                else:
                    left = [(int(lm_list[i].x * w), int(lm_list[i].y * h)) for i in LEFT_EYE]
                    right = [(int(lm_list[i].x * w), int(lm_list[i].y * h)) for i in RIGHT_EYE]
                    ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0

                    if frames_recovery > 0:
                        frames_recovery -= 1
                    if ear < EYE_AR_THRESH and frames_recovery == 0:
                        counter += 1
                        blink_detected = False
                    else:
                        if counter >= MIN_BLINK_FRAMES and not blink_detected:
                            blink_count += 1
                            blink_times.append(time.time())
                            blink_detected = True
                            frames_recovery = RECOVERY_FRAMES
                        counter = 0

                    blink_times = [t for t in blink_times if time.time() - t <= 60]
                    bpm = len(blink_times)

                    cv2.putText(frame, f"EAR: {ear:.2f}", (380, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Blink: {blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"Rate: {bpm}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                prev_landmarks = None
                counter = frames_recovery = 0
                cv2.putText(frame, "Face Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            _, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    finally:
        if cap: cap.release()

# Utility functions for routes
def reset_state():
    global blink_count, counter, blink_detected, frames_recovery, blink_times, prev_landmarks
    blink_count = 0
    counter = 0
    blink_detected = False
    frames_recovery = 0
    blink_times = []
    prev_landmarks = None

def get_blink_count():
    return blink_count

def get_face_status():
    return face_detected

def stop_camera():
    global detection_active
    detection_active = False
    time.sleep(0.2)

def start_camera():
    global detection_active
    detection_active = True
