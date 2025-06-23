import cv2
import time
import numpy as np
from scipy.spatial import distance
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EYE_AR_THRESH = 0.21
MIN_BLINK_FRAMES = 2
RECOVERY_FRAMES = 3
MOVEMENT_THRESH = 15

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

def process_frame(frame, state):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    face_detected = bool(results.multi_face_landmarks)

    if not face_detected:
        state['prev_landmarks'] = None
        state['counter'] = 0
        state['frames_recovery'] = 0
        return state['blink_count'], face_detected, state

    lm_list = results.multi_face_landmarks[0].landmark

    mv = detect_movement(lm_list, state['prev_landmarks']) if state['prev_landmarks'] else 0
    state['prev_landmarks'] = list(lm_list)

    if mv > MOVEMENT_THRESH:
        state['counter'] = 0
        state['blink_detected'] = False
        state['frames_recovery'] = 0
        return state['blink_count'], face_detected, state

    left = [(int(lm_list[i].x * w), int(lm_list[i].y * h)) for i in LEFT_EYE]
    right = [(int(lm_list[i].x * w), int(lm_list[i].y * h)) for i in RIGHT_EYE]
    ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0

    if state['frames_recovery'] > 0:
        state['frames_recovery'] -= 1

    if ear < EYE_AR_THRESH and state['frames_recovery'] == 0:
        state['counter'] += 1
        state['blink_detected'] = False
    else:
        if state['counter'] >= MIN_BLINK_FRAMES and not state['blink_detected']:
            state['blink_count'] += 1
            state['blink_times'].append(time.time())
            state['blink_detected'] = True
            state['frames_recovery'] = RECOVERY_FRAMES
        state['counter'] = 0

    state['blink_times'] = [t for t in state['blink_times'] if time.time() - t <= 60]
    return state['blink_count'], face_detected, state

def initial_state():
    return {
        'blink_count': 0,
        'counter': 0,
        'blink_detected': False,
        'frames_recovery': 0,
        'blink_times': [],
        'prev_landmarks': None
    }
