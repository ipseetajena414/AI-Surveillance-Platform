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
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===== GLOBAL STATE & SYNCHRONIZATION =====
lock = threading.Lock()

video_source = 0
cap = None
surveillance = None
camera_running = True

# Frame dimensions cache for client coordinate mapping
cached_frame_width = 750
cached_frame_height = 500

frame_stats = {
    'fps': 0.0,
    'latency_ms': 0.0,
    'cpu_percent': 0.0,
    'object_count': 0,
    'living_count': 0,
    'intrusion_active': False,
    'crowd_active': False,
    'model_type': 'YOLOv8 (CNN)',
    'camera_running': True,
    'frame_width': 750,
    'frame_height': 500
}


def init_camera(source=None):
    """
    Initialize or reinitialize video capture source.
    Thread-safe camera initializer.
    """
    global cap, video_source, camera_running, cached_frame_width, cached_frame_height
    with lock:
        if source is not None:
            video_source = source

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None

        src = int(video_source) if str(video_source).isdigit() else video_source
        new_cap = cv2.VideoCapture(src)

        if not new_cap.isOpened():
            print(f"[ERROR] Could not open video source: {video_source}")
            camera_running = False
            cap = None
            return False

        # Query frame dimensions
        w = int(new_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            cached_frame_width = w
            cached_frame_height = h
            frame_stats['frame_width'] = w
            frame_stats['frame_height'] = h

        cap = new_cap
        camera_running = True
        frame_stats['camera_running'] = True
        print(f"[OK] Camera / Media Source initialized: {video_source} ({cached_frame_width}x{cached_frame_height})")
        return True


def stop_camera():
    """
    Explicitly stop camera and release resources cleanly.
    """
    global cap, camera_running
    with lock:
        camera_running = False
        frame_stats['camera_running'] = False
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
        print("[OK] Camera released and stopped cleanly.")


def init_surveillance(model_path='yolov8n.pt', crowd_threshold=3):
    """Initialize the surveillance engine."""
    global surveillance
    with lock:
        if surveillance is None:
            surveillance = SurveillanceSystem(model_path=model_path, crowd_threshold=crowd_threshold)
        else:
            surveillance.load_model(model_path)
            surveillance.crowd_threshold = crowd_threshold
        print(f"[OK] Surveillance initialized ({surveillance.model_type}, threshold={crowd_threshold})")


def generate_frames():
    """MJPEG stream generator yielding annotated video frames."""
    global cap, surveillance, frame_stats, camera_running, cached_frame_width, cached_frame_height

    prev_time = time.time()

    while True:
        with lock:
            is_running = camera_running
            active_cap = cap

        # Standby / Stopped placeholder
        if not is_running or active_cap is None or not active_cap.isOpened():
            placeholder = np.zeros((cached_frame_height, cached_frame_width, 3), dtype=np.uint8)
            text = "CAMERA STOPPED - STANDBY" if not is_running else "INITIALIZING CAMERA FEED..."
            cv2.putText(placeholder, text, (max(20, int(cached_frame_width * 0.15)), int(cached_frame_height * 0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255) if not is_running else (150, 150, 150), 2)
            _, buffer = cv2.imencode('.jpg', placeholder)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.2)
            continue

        ret, frame = active_cap.read()
        if not ret:
            # Loop video files automatically
            with lock:
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.033)
            continue

        # Update cached dimensions
        h, w = frame.shape[:2]
        cached_frame_width = w
        cached_frame_height = h

        # Process frame through surveillance engine
        if surveillance:
            annotated_frame, intrusion, crowd = surveillance.process_frame(frame)
            latency = surveillance.last_latency_ms
            cpu_pct = surveillance.last_cpu_percent
            model_t = surveillance.model_type
            obj_cnt = surveillance._last_object_count
            living_cnt = surveillance._last_living_count
        else:
            annotated_frame = frame
            intrusion = False
            crowd = False
            latency = 0.0
            cpu_pct = 0.0
            model_t = 'None'
            obj_cnt = 0
            living_cnt = 0

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        with lock:
            frame_stats['fps'] = fps
            frame_stats['latency_ms'] = latency
            frame_stats['cpu_percent'] = cpu_pct
            frame_stats['intrusion_active'] = intrusion
            frame_stats['crowd_active'] = crowd
            frame_stats['model_type'] = model_t
            frame_stats['object_count'] = obj_cnt
            frame_stats['living_count'] = living_cnt
            frame_stats['frame_width'] = w
            frame_stats['frame_height'] = h

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


@app.route('/frame_size', methods=['GET'])
def get_frame_size():
    with lock:
        return jsonify({
            'width': cached_frame_width,
            'height': cached_frame_height,
            'camera_running': camera_running
        })


@app.route('/stop_camera', methods=['POST'])
def handle_stop_camera():
    stop_camera()
    return jsonify({'status': 'ok', 'camera_running': False})


@app.route('/start_camera', methods=['POST'])
def handle_start_camera():
    success = init_camera()
    return jsonify({'status': 'ok' if success else 'error', 'camera_running': camera_running})


@app.route('/set_roi', methods=['POST'])
def set_roi():
    data = request.get_json() or {}
    points = data.get('points', [])
    zone_id = data.get('zone_id', 'zone_1')
    zone_name = data.get('zone_name', 'Red Zone 1')

    if len(points) < 3:
        return jsonify({'status': 'error', 'message': 'At least 3 points required for a polygon'}), 400

    if surveillance:
        surveillance.set_zone(zone_id=zone_id, zone_name=zone_name, points=points)

    print(f"[ROI] Configured '{zone_name}' ({zone_id}) with {len(points)} nodes: {points}")
    return jsonify({'status': 'ok', 'points': len(points), 'zone_id': zone_id, 'zone_name': zone_name})


@app.route('/clear_roi', methods=['POST'])
def clear_roi():
    data = request.get_json(silent=True) or {}
    zone_id = data.get('zone_id', None)

    if surveillance:
        surveillance.clear_zone(zone_id=zone_id)

    print(f"[ROI] Cleared {'zone ' + zone_id if zone_id else 'all zones'}")
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
    zones_info = surveillance.get_zones_info() if surveillance else []

    stats['alerts'] = alerts
    stats['breach_history'] = breach_history
    stats['total_alerts_count'] = getattr(surveillance, 'total_alerts_count', len(alerts)) if surveillance else 0
    stats['total_breaches'] = getattr(surveillance, 'total_alerts_count', len(breach_history)) if surveillance else 0
    stats['zones'] = zones_info
    stats['object_count'] = getattr(surveillance, '_last_object_count', 0) if surveillance else 0
    stats['living_count'] = getattr(surveillance, '_last_living_count', 0) if surveillance else 0

    return jsonify(stats)


@app.route('/clear_alerts', methods=['POST'])
def clear_alerts():
    if surveillance:
        surveillance.recent_alerts = []
        surveillance.breach_history = []
        surveillance.reset_roi_tracking()
    return jsonify({'status': 'ok'})


@app.route('/config', methods=['POST'])
def config():
    global video_source
    data = request.get_json() or {}

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
