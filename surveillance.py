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
        self.recent_alerts = []  
        
        # ID Tracking & Red Zone Breach Registry
        self.logged_breach_ids = set()  # Set of (obj_id, class_name) already logged for current ROI
        self.breach_history = []        # List of dicts: {'id': obj_id, 'class': class_name, 'time': timestamp, 'type': 'Vehicle'/'Living'}
        
        self.model_type = "YOLO"
        self.model_path = model_path
        
        self.load_model(model_path)

        # Performance metrics
        self.last_latency_ms = 0.0
        self.last_cpu_percent = 0.0
        self._last_object_count = 0
        self._last_living_count = 0
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("AI Surveillance System Log\n")
                f.write("="*30 + "\n")

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

    def reset_roi_tracking(self):
        """Reset logged IDs when ROI changes or is cleared."""
        self.logged_breach_ids.clear()
        self.breach_history.clear()

    def log_id_breach(self, obj_id, class_name):
        """Record unique Car ID or Living Being ID passing through the Red Zone."""
        key = (str(obj_id), class_name.lower())
        if key in self.logged_breach_ids:
            return  # Already logged this object ID for this red zone

        self.logged_breach_ids.add(key)
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        
        category = "Living" if class_name.lower() in self.living_classes else "Vehicle"
        display_id = f"{class_name} #{obj_id}" if obj_id != "N/A" else f"{class_name}"

        entry = {
            'id': display_id,
            'class': class_name,
            'category': category,
            'time': timestamp,
            'raw_id': str(obj_id)
        }
        self.breach_history.append(entry)

        # File logging
        log_msg = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚨 RED ZONE BREACH: {display_id} ({category}) entered at {timestamp}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

        self.recent_alerts.append(log_msg)
        if len(self.recent_alerts) > 50:
            self.recent_alerts = self.recent_alerts[-50:]

    def play_alarm(self):
        """Play audio alert (Windows winsound)."""
        if winsound:
            winsound.Beep(1000, 200)

    def process_frame(self, frame, roi_polygon=None):
        """
        Process frame, track objects with IDs, check red zone breaches, and record ID logs.
        """
        start_inference = time.time()
        h, w = frame.shape[:2]
        annotated_frame = frame.copy()
        
        intrusion_detected = False
        object_count = 0
        living_count = 0
        
        # Draw ROI polygon as RED ZONE if defined
        if roi_polygon is not None and len(roi_polygon) > 2:
            pts = np.array(roi_polygon, np.int32).reshape((-1, 1, 2))
            
            # Semi-transparent Red Zone fill + bright Red border
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.25, annotated_frame, 0.75, 0, annotated_frame)
            cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
            cv2.putText(annotated_frame, "🚨 RED ZONE (RESTRICTED AREA)", (pts[0][0][0], max(25, pts[0][0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

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

        # Process detected objects & check Red Zone containment
        for idx, (box, class_name, conf, obj_id) in enumerate(detected_objects):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int(y2)  # Bottom anchor verification point
            
            if class_name.lower() in self.living_classes:
                living_count += 1
                
            in_roi = False
            if roi_polygon is not None and len(roi_polygon) > 2:
                in_roi = is_inside_polygon(cx, cy, roi_polygon)
                if in_roi:
                    intrusion_detected = True
                    # Record specific Car ID / Living Being ID into breach history log
                    self.log_id_breach(obj_id, class_name)

            color = (0, 0, 255) if in_roi else (0, 255, 0)
            id_tag = f" ID:{obj_id}" if obj_id != "N/A" else ""
            label_str = f"{'🚨 RED ZONE' if in_roi else class_name}{id_tag} ({conf:.2f})"

            # Draw bounding box & centroid
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated_frame, (cx, cy), 5, color, -1)
            cv2.putText(annotated_frame, label_str, (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Trigger Audio Alarm
        if intrusion_detected:
            cv2.putText(annotated_frame, "⚠️ RED ZONE BREACH DETECTED!", (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if living_count >= self.crowd_threshold:
            cv2.putText(annotated_frame, "🧑‍🤝‍🧑 CROWD DENSITY ALERT!", (40, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

        if intrusion_detected or living_count >= self.crowd_threshold:
            current_time = time.time()
            if current_time - self.last_alarm_time > 1.5:
                self.last_alarm_time = current_time
                threading.Thread(target=self.play_alarm, daemon=True).start()

        self._last_object_count = object_count
        self._last_living_count = living_count

        return annotated_frame, intrusion_detected, living_count >= self.crowd_threshold
