from flask import render_template, Response, jsonify
from detection import generate, reset_state, get_blink_count, get_face_status, stop_camera, start_camera

def configure_routes(app):
    # In routes.py
    @app.route('/')
    def index():
        return jsonify({"message": "Blink Detection Backend is running"})


    @app.route('/video_feed')
    def video_feed():
        start_camera()
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/stop')
    def stop():
        stop_camera()
        return jsonify({'status': 'stopped'})

    @app.route('/blink_count')
    def blink_count_api():
        return jsonify({'blink_count': get_blink_count()})

    @app.route('/reset')
    def reset():
        reset_state()
        return jsonify({'status': 'reset'})

    @app.route('/face_status')
    def face_status():
        return jsonify({'face_detected': get_face_status()})

    @app.route('/health')
    def health():
        return jsonify({'status': 'ok'}), 200
