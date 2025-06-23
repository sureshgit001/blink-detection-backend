from flask import request, jsonify
import base64
import numpy as np
import cv2
from detection import process_frame, initial_state

client_states = {}

def register_routes(app):

    @app.route('/upload_frame', methods=['POST'])
    def upload_frame():
        try:
            data = request.get_json()
            client_id = data.get('client_id')
            if not client_id:
                return jsonify({'error': 'Missing client_id'}), 400

            if client_id not in client_states:
                client_states[client_id] = initial_state()

            img_data = data['image'].split(',')[1]
            img_bytes = base64.b64decode(img_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            count, face, updated_state = process_frame(frame, client_states[client_id])
            client_states[client_id] = updated_state
            return jsonify({"blink_count": count, "face_detected": face})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


    @app.route('/reset', methods=['POST'])
    def reset():
        data = request.get_json()
        client_id = data.get('client_id')
        if not client_id:
            return jsonify({'error': 'Missing client_id'}), 400

        client_states[client_id] = initial_state()
        return jsonify({'status': 'reset'})

    @app.route('/health')
    def health():
        return jsonify({'status': 'ok'}), 200
