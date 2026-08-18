"""
AI Multi-Model Surveillance Engine with Red Zone Overlap Detection.

Core detection engine supporting:
- YOLOv8 persistent object tracking with unique IDs
- DETR (Transformer) fallback detection
- Multi-zone bounding-box / polygon overlap detection (configurable threshold)
- Per-object ENTER/INSIDE/EXIT state machine with 1-second alert throttling
- Crowd density monitoring
- Throttled file logging and asynchronous audio alarms
"""

import cv2
import numpy as np
import datetime
import time
import os
import threading
import platform
import psutil
from PIL import Image

# Fix OpenMP runtime duplicate error on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from ultralytics import YOLO

# Optional DETR imports
try:
    from transformers import DetrForObjectDetection, DetrImageProcessor
    HAS_DETR = True
except ImportError:
    HAS_DETR = False

# Cross-platform audio support
if platform.system() == 'Windows':
    import winsound
else:
    winsound = None

# ===== CONFIGURABLE THRESHOLDS =====
RED_ZONE_OVERLAP_THRESHOLD = 0.05   # 5% bounding-box overlap triggers alert
ALERT_INTERVAL = 1.0                # Seconds between repeated alerts per object per zone
MAX_RECENT_ALERTS = 500             # Maximum alerts kept in memory
MAX_LOG_ALERTS_PER_SECOND = 10      # Maximum file log writes per second (global)


def calculate_bbox_polygon_overlap(bbox, polygon_mask, frame_shape):
    """
    Calculate what fraction of the bounding box area overlaps the polygon.
    Uses OpenCV binary mask intersection — efficient and dependency-free.

    Args:
        bbox: (x1, y1, x2, y2) bounding box coordinates
        polygon_mask: Pre-computed binary mask for the polygon (np.uint8, 255=inside)
        frame_shape: (height, width) of the frame

    Returns:
        float: overlap ratio (0.0 to 1.0)
    """
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]

    # Clamp to frame boundaries
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_pixels = bbox_width * bbox_height

    if bbox_pixels <= 0:
        return 0.0

    # Extract the polygon mask region corresponding to the bbox
    roi = polygon_mask[y1:y2, x1:x2]
    intersection_pixels = cv2.countNonZero(roi)

    return intersection_pixels / bbox_pixels


def build_polygon_mask(polygon_pts, frame_shape):
    """
    Create a binary mask for a polygon. Called once per frame per zone.

    Args:
        polygon_pts: list of (x, y) tuples
        frame_shape: (height, width, ...) of the frame

    Returns:
        np.ndarray: binary mask (uint8, 255=inside polygon)
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon_pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


# Legacy function kept for backward compatibility (main.py uses it)
def is_inside_polygon(x, y, poly):
    """
    Ray-casting algorithm engine to determine if a point (x, y) is inside a polygon.
    As documented in Section 6.1 (Equation 1) of the technical report.
    """
    if not poly or len(poly) < 3:
        return False
    num = len(poly)
    j = num - 1
    c = False
    for i in range(num):
        if ((poly[i][1] > y) != (poly[j][1] > y)) and \
                (x < (poly[j][0] - poly[i][0]) * (y - poly[i][1]) / (poly[j][1] - poly[i][1] + 1e-6) + poly[i][0]):
            c = not c
        j = i
    return c


class SurveillanceSystem:
    def __init__(self, model_path='yolov8n.pt', crowd_threshold=3, log_file='alerts.log'):
        """
        Initialize the Multi-Model Surveillance System with Object ID Tracking.
        """
        self.crowd_threshold = crowd_threshold
        self.log_file = log_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.living_classes = ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
        self.vehicle_classes = ['car', 'motorcycle', 'bus', 'truck', 'bicycle', 'train', 'airplane', 'boat']

        self.last_alarm_time = 0
        self.total_alerts_count = 0
        self.recent_alerts = []         # List of alert dicts (enhanced data model)

        # Multi-zone support: list of {'id': str, 'name': str, 'points': [(x,y), ...]}
        self.zones = []

        # Per-object per-zone state machine: key = (zone_id, str(obj_id))
        # value = {'state': 'INSIDE'|'OUTSIDE', 'last_alert_time': float,
        #          'overlap_percent': float, 'class_name': str, 'confidence': float}
        self.object_zone_states = {}

        # Legacy breach history for backward compat with existing UI
        self.breach_history = []

        self.model_type = "YOLO"
        self.model_path = model_path

        self.load_model(model_path)

        # Performance metrics
        self.last_latency_ms = 0.0
        self.last_cpu_percent = 0.0
        self._last_object_count = 0
        self._last_living_count = 0

        # File log rate limiter
        self._last_file_log_time = 0.0

        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("AI Surveillance System Log\n")
                f.write("=" * 30 + "\n")

    def load_model(self, model_path):
        """Switch or load AI detection model (YOLOv8 or DETR)."""
        print(f"Loading AI Model: {model_path} on {self.device}...")
        self.model_path = model_path

        if "detr" in model_path.lower() or "transformer" in model_path.lower():
            if not HAS_DETR:
                print("[WARNING] transformers library not available. Falling back to YOLOv8.")
                self.model = YOLO('yolov8n.pt')
                self.model_type = "YOLOv8 (CNN)"
            else:
                self.model_type = "DETR (Transformer)"
                model_name = "facebook/detr-resnet-50"
                self.detr_processor = DetrImageProcessor.from_pretrained(model_name)
                self.detr_model = DetrForObjectDetection.from_pretrained(model_name).to(self.device)
                self.detr_model.eval()
                self.detr_classes = {1: "Person", 3: "Car", 4: "Motorcycle", 6: "Bus", 8: "Truck"}
        else:
            self.model_type = "YOLOv8 (CNN)"
            self.model = YOLO(model_path)
            self.model.to(self.device)

    # ===== ZONE MANAGEMENT =====

    def set_zone(self, zone_id, zone_name, points):
        """Add or update a red zone polygon."""
        # Remove existing zone with same id
        self.zones = [z for z in self.zones if z['id'] != zone_id]
        self.zones.append({
            'id': zone_id,
            'name': zone_name,
            'points': [(int(p[0]), int(p[1])) for p in points]
        })
        # Clear states for this zone so tracking restarts
        keys_to_remove = [k for k in self.object_zone_states if k[0] == zone_id]
        for k in keys_to_remove:
            del self.object_zone_states[k]
        print(f"[ZONE] Set zone '{zone_name}' ({zone_id}) with {len(points)} points")

    def clear_zone(self, zone_id=None):
        """Clear a specific zone or all zones."""
        if zone_id:
            self.zones = [z for z in self.zones if z['id'] != zone_id]
            keys_to_remove = [k for k in self.object_zone_states if k[0] == zone_id]
            for k in keys_to_remove:
                del self.object_zone_states[k]
            print(f"[ZONE] Cleared zone {zone_id}")
        else:
            self.zones.clear()
            self.object_zone_states.clear()
            print("[ZONE] Cleared all zones")

    def reset_roi_tracking(self):
        """Reset all tracking states. Called when zones change significantly."""
        self.object_zone_states.clear()
        self.breach_history.clear()

    def get_zones_info(self):
        """Return zone info for API responses."""
        return [{'id': z['id'], 'name': z['name'], 'point_count': len(z['points'])} for z in self.zones]

    # ===== ALERT SYSTEM =====

    def _create_alert(self, obj_id, class_name, confidence, zone_id, zone_name, overlap_percent, event):
        """Create an alert entry and manage storage."""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        self.total_alerts_count += 1

        category = "Living" if class_name.lower() in self.living_classes else "Vehicle"
        display_id = f"{class_name} #{obj_id}" if obj_id != "N/A" else class_name

        alert = {
            'id': self.total_alerts_count,
            'timestamp': timestamp,
            'object_id': str(obj_id),
            'class': class_name,
            'display_id': display_id,
            'category': category,
            'confidence': round(confidence, 2),
            'zone_id': zone_id,
            'zone_name': zone_name,
            'overlap_percent': round(overlap_percent * 100, 1),
            'event': event   # 'ENTER', 'INSIDE', 'EXIT'
        }

        self.recent_alerts.append(alert)
        if len(self.recent_alerts) > MAX_RECENT_ALERTS:
            self.recent_alerts = self.recent_alerts[-MAX_RECENT_ALERTS:]

        # Also add to breach_history for backward compat
        if event in ('ENTER', 'INSIDE'):
            breach_entry = {
                'id': display_id,
                'class': class_name,
                'category': category,
                'time': timestamp,
                'raw_id': str(obj_id),
                'zone_id': zone_id,
                'zone_name': zone_name,
                'overlap_percent': round(overlap_percent * 100, 1),
                'event': event
            }
            self.breach_history.append(breach_entry)
            if len(self.breach_history) > MAX_RECENT_ALERTS:
                self.breach_history = self.breach_history[-MAX_RECENT_ALERTS:]

        # Throttled file logging
        current_time = time.time()
        if current_time - self._last_file_log_time >= (1.0 / MAX_LOG_ALERTS_PER_SECOND):
            self._last_file_log_time = current_time
            icon = "ALERT" if event != "EXIT" else "CLEAR"
            log_msg = (f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                       f"{icon} {event}: {display_id} | Zone: {zone_name} | "
                       f"Overlap: {round(overlap_percent * 100, 1)}% | Conf: {round(confidence, 2)}")
            try:
                with open(self.log_file, 'a') as f:
                    f.write(log_msg + "\n")
            except Exception:
                pass

    def play_alarm(self):
        """Play audio alert (Windows winsound)."""
        if winsound:
            try:
                winsound.Beep(1000, 200)
            except Exception:
                pass

    # ===== MAIN PROCESSING =====

    def process_frame(self, frame, roi_polygon=None):
        """
        Process frame, track objects with IDs, check red zone overlap, generate alerts.

        Args:
            frame: BGR numpy array from camera
            roi_polygon: Legacy single polygon support (list of (x,y) tuples).
                        If provided and no zones exist, auto-creates 'zone_1'.

        Returns:
            (annotated_frame, intrusion_detected, crowd_alert)
        """
        start_inference = time.time()
        h, w = frame.shape[:2]
        annotated_frame = frame.copy()

        intrusion_detected = False
        object_count = 0
        living_count = 0
        current_time = time.time()

        # Legacy compatibility: if roi_polygon passed and no zones set, create one
        if roi_polygon and len(roi_polygon) > 2 and len(self.zones) == 0:
            self.set_zone('zone_1', 'Red Zone 1', roi_polygon)

        # ===== DRAW ALL ZONE POLYGONS =====
        for zone in self.zones:
            pts = np.array(zone['points'], np.int32).reshape((-1, 1, 2))

            # Semi-transparent Red Zone fill + bright Red border
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.20, annotated_frame, 0.80, 0, annotated_frame)
            cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 0, 255), thickness=3)

            # Zone label
            label_pt = (pts[0][0][0], max(25, pts[0][0][1] - 10))
            cv2.putText(annotated_frame, f"RED ZONE ({zone['name']})",
                        label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw corner nodes
            for pt in zone['points']:
                cv2.circle(annotated_frame, pt, 4, (0, 100, 255), -1)

        # ===== BUILD POLYGON MASKS (once per frame per zone) =====
        zone_masks = {}
        for zone in self.zones:
            if len(zone['points']) >= 3:
                zone_masks[zone['id']] = build_polygon_mask(zone['points'], (h, w))

        # ===== OBJECT DETECTION =====
        detected_objects = []

        if "DETR" in self.model_type:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.detr_processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.detr_model(**inputs)

            results = self.detr_processor.post_process_object_detection(
                outputs, target_sizes=torch.tensor([[h, w]]).to(self.device), threshold=0.25
            )[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                lbl_id = label.item()
                class_name = self.detr_classes.get(lbl_id, None)
                if class_name:
                    box_coords = [int(i) for i in box.tolist()]
                    # Generate pseudo ID based on position for DETR
                    pseudo_id = f"{box_coords[0] // 20}"
                    detected_objects.append((box_coords, class_name, score.item(), pseudo_id))
        else:
            # YOLOv8 Tracking with Persistent IDs
            yolo_results = self.model.track(frame, persist=True, verbose=False)[0]
            if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
                boxes = yolo_results.boxes.xyxy.cpu().numpy()
                confs = yolo_results.boxes.conf.cpu().numpy()
                class_ids = yolo_results.boxes.cls.cpu().numpy()
                ids = yolo_results.boxes.id.cpu().numpy() if yolo_results.boxes.id is not None else [None] * len(boxes)

                for i, box in enumerate(boxes):
                    cls_id = int(class_ids[i])
                    class_name = self.model.names[cls_id].capitalize()
                    obj_id = int(ids[i]) if ids[i] is not None else "N/A"
                    detected_objects.append(([int(b) for b in box], class_name, float(confs[i]), obj_id))

        end_inference = time.time()
        self.last_latency_ms = (end_inference - start_inference) * 1000.0
        self.last_cpu_percent = psutil.cpu_percent()

        object_count = len(detected_objects)

        # Track which (zone_id, obj_id) keys are active this frame
        active_keys = set()

        # ===== PROCESS EACH OBJECT AGAINST EACH ZONE =====
        for idx, (box, class_name, conf, obj_id) in enumerate(detected_objects):
            x1, y1, x2, y2 = box

            if class_name.lower() in self.living_classes:
                living_count += 1

            # Check overlap against each zone
            obj_in_any_zone = False

            for zone in self.zones:
                zone_id = zone['id']
                mask = zone_masks.get(zone_id)
                if mask is None:
                    continue

                overlap = calculate_bbox_polygon_overlap((x1, y1, x2, y2), mask, (h, w))

                state_key = (zone_id, str(obj_id))
                active_keys.add(state_key)

                prev_state = self.object_zone_states.get(state_key, {
                    'state': 'OUTSIDE',
                    'last_alert_time': 0.0,
                    'overlap_percent': 0.0,
                    'class_name': class_name,
                    'confidence': conf
                })

                if overlap >= RED_ZONE_OVERLAP_THRESHOLD:
                    obj_in_any_zone = True
                    intrusion_detected = True

                    if prev_state['state'] == 'OUTSIDE':
                        # ENTER event
                        self._create_alert(obj_id, class_name, conf, zone_id,
                                           zone['name'], overlap, 'ENTER')
                        prev_state['state'] = 'INSIDE'
                        prev_state['last_alert_time'] = current_time
                    elif current_time - prev_state['last_alert_time'] >= ALERT_INTERVAL:
                        # Periodic INSIDE alert (~1 per second)
                        self._create_alert(obj_id, class_name, conf, zone_id,
                                           zone['name'], overlap, 'INSIDE')
                        prev_state['last_alert_time'] = current_time

                    prev_state['overlap_percent'] = overlap
                    prev_state['confidence'] = conf
                    prev_state['class_name'] = class_name
                else:
                    if prev_state['state'] == 'INSIDE':
                        # EXIT event (single log)
                        self._create_alert(obj_id, class_name, conf, zone_id,
                                           zone['name'], overlap, 'EXIT')
                        prev_state['state'] = 'OUTSIDE'
                        prev_state['last_alert_time'] = 0.0

                self.object_zone_states[state_key] = prev_state

            # ===== DRAW BOUNDING BOX =====
            color = (0, 0, 255) if obj_in_any_zone else (0, 255, 0)
            id_tag = f" ID:{obj_id}" if obj_id != "N/A" else ""

            if obj_in_any_zone:
                # Find the max overlap for display
                max_overlap = 0.0
                for zone in self.zones:
                    sk = (zone['id'], str(obj_id))
                    st = self.object_zone_states.get(sk, {})
                    ov = st.get('overlap_percent', 0.0)
                    if ov > max_overlap:
                        max_overlap = ov

                label_str = f"RED ZONE | {class_name}{id_tag} | {conf:.0%} | {max_overlap * 100:.0f}%"

                # Thicker red bounding box for violations
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                # Alert label background
                (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated_frame, (x1, max(0, y1 - th - 8)),
                              (x1 + tw + 6, max(0, y1 - 2)), (0, 0, 200), -1)
                cv2.putText(annotated_frame, label_str, (x1 + 3, max(th + 2, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            else:
                label_str = f"{class_name}{id_tag} ({conf:.2f})"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label_str, (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ===== PRUNE STALE OBJECT STATES =====
        # Remove entries for objects no longer detected
        stale_keys = [k for k in self.object_zone_states if k not in active_keys]
        for k in stale_keys:
            state = self.object_zone_states[k]
            if state['state'] == 'INSIDE':
                # Object disappeared while inside — generate EXIT
                zone_info = next((z for z in self.zones if z['id'] == k[0]), None)
                if zone_info:
                    self._create_alert(k[1], state['class_name'], state['confidence'],
                                       k[0], zone_info['name'], 0.0, 'EXIT')
            del self.object_zone_states[k]

        # ===== GLOBAL ALERT OVERLAYS =====
        if intrusion_detected:
            cv2.putText(annotated_frame, "RED ZONE BREACH DETECTED!", (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if living_count >= self.crowd_threshold:
            cv2.putText(annotated_frame, "CROWD DENSITY ALERT!", (40, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

        # Audio alarm (throttled)
        if intrusion_detected or living_count >= self.crowd_threshold:
            if current_time - self.last_alarm_time > 1.5:
                self.last_alarm_time = current_time
                threading.Thread(target=self.play_alarm, daemon=True).start()

        self._last_object_count = object_count
        self._last_living_count = living_count

        return annotated_frame, intrusion_detected, living_count >= self.crowd_threshold
