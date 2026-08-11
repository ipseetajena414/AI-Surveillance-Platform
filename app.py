"""
AI Multi-Model Surveillance & Red Zone ID Tracking Platform
Flask Web Application backend serving MJPEG video feed, Red Zone ID breach tracking, and real-time report status.
"""

import os
import cv2
import time
import argparse
import threading
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename
from surveillance import SurveillanceSystem

# Fix OpenMP runtime duplicate error on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===== GLOBAL STATE =====
lock = threading.Lock()

video_source = 0
cap = None
surveillance = None
roi_polygon = []

frame_stats = {
    'fps': 0.0,
    'latency_ms': 0.0,
    'cpu_percent': 0.0,
    'object_count': 0,
    'living_count': 0,
    'intrusion_active': False,
    'crowd_active': False,
    'model_type': 'YOLOv8 (CNN)'
}


def init_camera(source):
    """Initialize or reinitialize video capture source."""
    global cap
    if cap is not None:
        cap.release()
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return False
    print(f"[OK] Camera / Media Source initialized: {source}")
    return True


def init_surveillance(model_path='yolov8n.pt', crowd_threshold=3):
    """Initialize the surveillance engine."""
    global surveillance
    if surveillance is None:
        surveillance = SurveillanceSystem(model_path=model_path, crowd_threshold=crowd_threshold)
    else:
        surveillance.load_model(model_path)
        surveillance.crowd_threshold = crowd_threshold
    print(f"[OK] Surveillance initialized ({surveillance.model_type}, threshold={crowd_threshold})")


def generate_frames():
    """MJPEG stream generator yielding annotated video frames."""
    global cap, surveillance, roi_polygon, frame_stats

    prev_time = time.time()

    while True:
        if cap is None or not cap.isOpened():
            placeholder = np.zeros((500, 750, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No Media Input Feed Active", (180, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2)
            _, buffer = cv2.imencode('.jpg', placeholder)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.3)
            continue

        ret, frame = cap.read()
        if not ret:
            # Loop video files automatically
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.033)
            continue

        with lock:
            current_roi = list(roi_polygon)

        roi_tuples = [(int(p[0]), int(p[1])) for p in current_roi] if current_roi else None

        annotated_frame, intrusion, crowd = surveillance.process_frame(frame, roi_polygon=roi_tuples)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        with lock:
            frame_stats['fps'] = fps
            frame_stats['latency_ms'] = surveillance.last_latency_ms
            frame_stats['cpu_percent'] = surveillance.last_cpu_percent
            frame_stats['intrusion_active'] = intrusion
            frame_stats['crowd_active'] = crowd
            frame_stats['model_type'] = surveillance.model_type

        # Encode to JPEG for browser MJPEG streaming
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ===== ROUTES =====

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_roi', methods=['POST'])
def set_roi():
    global roi_polygon
    data = request.get_json()
    points = data.get('points', [])
    with lock:
        roi_polygon = points
    if surveillance:
        surveillance.reset_roi_tracking()
    print(f"[ROI] Defined Red Zone polygon with {len(points)} nodes: {points}")
    return jsonify({'status': 'ok', 'points': len(points)})


@app.route('/clear_roi', methods=['POST'])
def clear_roi():
    global roi_polygon
    with lock:
        roi_polygon = []
    if surveillance:
        surveillance.reset_roi_tracking()
    print("[ROI] Cleared")
    return jsonify({'status': 'ok'})


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload media file (video/image) and set it as active input source."""
    global video_source
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    video_source = filepath
    init_camera(video_source)
    if surveillance:
        surveillance.reset_roi_tracking()
    return jsonify({'status': 'ok', 'filename': filename, 'path': filepath})


@app.route('/status', methods=['GET'])
def status():
    with lock:
        stats = dict(frame_stats)

    alerts = surveillance.recent_alerts if surveillance else []
    breach_history = surveillance.breach_history if surveillance else []
    
    stats['alerts'] = alerts
    stats['breach_history'] = breach_history
    stats['total_breaches'] = len(breach_history)
    stats['object_count'] = getattr(surveillance, '_last_object_count', 0) if surveillance else 0
    stats['living_count'] = getattr(surveillance, '_last_living_count', 0) if surveillance else 0

    return jsonify(stats)


@app.route('/clear_alerts', methods=['POST'])
def clear_alerts():
    if surveillance:
        surveillance.recent_alerts = []
        surveillance.reset_roi_tracking()
    return jsonify({'status': 'ok'})


@app.route('/config', methods=['POST'])
def config():
    global video_source
    data = request.get_json()

    if 'crowd_threshold' in data:
        if surveillance:
            surveillance.crowd_threshold = int(data['crowd_threshold'])

    if 'source' in data:
        new_source = data['source']
        if str(new_source) != str(video_source):
            video_source = new_source
            init_camera(video_source)
            if surveillance:
                surveillance.reset_roi_tracking()

    if 'model' in data:
        new_model = data['model']
        threshold = surveillance.crowd_threshold if surveillance else 3
        init_surveillance(model_path=new_model, crowd_threshold=threshold)

    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Model AI Surveillance & Red Zone ID Tracking Platform")
    parser.add_argument('--source', type=str, default='0', help='Video source: 0 for webcam, or path to file')
    parser.add_argument('--crowd_threshold', type=int, default=3, help='Crowd density alert threshold')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model path (yolov8n.pt or detr)')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host IP')
    args = parser.parse_args()

    video_source = args.source
    init_camera(video_source)
    init_surveillance(model_path=args.model, crowd_threshold=args.crowd_threshold)

    print(f"\n{'='*60}")
    print(f" 📹 AI Multi-Model Surveillance & Red Zone ID Tracker")
    print(f" 🌐 Dashboard available at: http://localhost:{args.port}")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
