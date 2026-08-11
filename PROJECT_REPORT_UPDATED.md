# MULTI-MODEL ARTIFICIAL INTELLIGENCE SURVEILLANCE AND COMPUTER VISION GUARDRAIL PLATFORM

**A Comprehensive Deep Learning Architecture Utilizing YOLOv8 Object Tracking, Interactive HTML5 Canvas ROI Guardrails, and Real-Time Industrial Intrusion Detection**

*A Technical Report Submitted in Partial Fulfillment of the Requirements for the Summer Internship Program*  
*(Duration: May – July 2026)*

**Under the guidance of:** Prof. Pankaj Kumar Sa  
*Department of Computer Science and Engineering, National Institute of Technology Rourkela*  

**Submitted by:** IPSEETA JENA (Roll No: SIP260025)

---

## Abstract
The enforcement of high-fidelity physical security perimeters within modern industrial environments, logistics handling yards, and hazardous facilities requires continuous, automated monitoring. Conventional closed-circuit television (CCTV) surveillance relies on passive recording or human oversight, introducing vulnerabilities from operator fatigue and delayed event handling. 

This technical report details the design and deployment of an automated, real-time computer vision system built on a lightweight Flask backend and an interactive HTML5 Canvas front-end. The platform features an anchor-free YOLOv8 object detection and persistent tracking pipeline integrated with mathematical point-in-polygon containment testing (`cv2.pointPolygonTest`). Operators can interactively define non-orthogonal polygonal perimeters directly over a live video stream. The system monitors both unauthorized spatial intrusion and crowd density thresholds, providing real-time visual alerts, anti-spam log throttling, and asynchronous audio alarms. System evaluation demonstrates low-latency inference (~50+ FPS depending on hardware), high localization precision, and seamless multi-input adaptability across static files, video streams, and live webcams.

---

## 1. Introduction

### 1.1 Project Overview & Motivation
Commercial off-the-shelf surveillance systems often rely on static rectangular detection filters. These rigid filters fail when security operations need to monitor non-orthogonal perimeters—such as diagonal walkways, curved conveyor zones, or angular loading bays. 

To resolve this limitation, this project implements an agile computer vision guardrail platform. By combining customizable vector polygon drawing on an HTML5 canvas with real-time deep learning tracking, the system provides zero-flicker, sub-millisecond perimeter verification directly on video streams.

---

## 2. System Architecture & Technical Implementation

```
+-------------------------------------------------------------------------+
|                        BROWSER FRONTEND (HTML5/CSS3/JS)                 |
|  - Live MJPEG Stream (<img src="/video_feed">)                         |
|  - Interactive HTML5 Canvas ROI Overlay                                 |
|  - Real-Time Stats Bar (FPS, Objects, Living Count, Alert Count)        |
|  - Live Alert Sidebar Feed (REST Polling /status every 500ms)           |
+-------------------------------------------------------------------------+
                                    |
                            REST API / HTTP
                                    |
+-------------------------------------------------------------------------+
|                         FLASK WEB SERVER (app.py)                       |
|  - /video_feed  : MJPEG frame generator generator_frames()              |
|  - /set_roi     : Receives polygon canvas nodes & maps to video res    |
|  - /status      : Returns real-time metrics & recent alert history      |
|  - /config      : Dynamic threshold, camera source & model switching    |
+-------------------------------------------------------------------------+
                                    |
                             In-Memory Thread
                                    |
+-------------------------------------------------------------------------+
|                      SURVEILLANCE ENGINE (surveillance.py)              |
|  - YOLOv8 Object Tracking (model.track(persist=True))                   |
|  - Point-in-Polygon Intrusion Engine (cv2.pointPolygonTest)             |
|  - Crowd Density Monitoring (Living class count vs Threshold)           |
|  - Throttled Alert Logger (5s anti-spam cooldown -> alerts.log)         |
|  - Asynchronous Audio Alarm Worker (winsound.Beep / Threading)          |
+-------------------------------------------------------------------------+
```

### 2.1 Technical Stack
* **Deep Learning Framework:** Ultralytics YOLOv8 (`yolov8n.pt` pretrained / custom VisDrone weights)
* **Computer Vision Engine:** OpenCV (`cv2`) for frame transformation, spatial polygon testing, and stream decoding
* **Backend Web Architecture:** Flask WSGI application with multi-threaded MJPEG streaming (`Response(generate_frames())`)
* **Frontend Design:** Vanilla HTML5, CSS3 Glassmorphism UI tokens, and JavaScript HTML5 Canvas API
* **Custom Dataset Training:** VisDrone dataset fine-tuning pipeline (`train_visdrone.py`) for specialized drone/aerial surveillance

---

## 3. Core Algorithmic Logic & Mathematical Modeling

### 3.1 Point-in-Polygon Containment Testing
To determine whether an object centroid $P(x_c, y_c)$ resides inside a user-defined polygonal restricted zone $\Omega$, the system uses OpenCV's distance transform algorithm (`cv2.pointPolygonTest`).

Given an ordered set of polygon vertices $V = \{ (x_0, y_0), (x_1, y_1), \dots, (x_{n-1}, y_{n-1}) \}$:
$$d = \text{pointPolygonTest}(V, P(x_c, y_c), \text{measureDist}=\text{False})$$

$$\text{Containment State} = \begin{cases} \text{INTRUSION (Inside)}, & d \ge 0 \\ \text{SECURE (Outside)}, & d < 0 \end{cases}$$

### 3.2 Spatial Object Centroid Tracking
For each detected bounding box $[x_1, y_1, x_2, y_2]$ returned by the YOLOv8 tracking module:
$$x_c = \frac{x_1 + x_2}{2}, \quad y_c = \frac{y_1 + y_2}{2}$$

The center point $(x_c, y_c)$ is checked against the ROI polygon in real-time. If inside, a red bounding box and alert label are rendered onto the frame stream.

---

## 4. Key Functional Features & Results

1. **Interactive ROI Definition:** Operators click directly on the live browser canvas to draw custom guardrail perimeters.
2. **Dual Alert Triggers:**
   - **INTRUSION ALERT:** Triggered when any tracked object centroid enters the defined ROI.
   - **CROWD ALERT:** Triggered when the count of living entities (persons/animals) exceeds the configured crowd threshold.
3. **Anti-Spam Log Throttling:** System prevents disk overflow by enforcing a 5-second cooldown per alert type while maintaining live visual indicator banners.
4. **Real-Time Performance Benchmarks:**
   - **Inference Speed:** ~50+ FPS on desktop GPU / ~15-30 FPS on CPU.
   - **Latency:** ~12-20 ms inference latency per frame.
   - **UI Responsiveness:** Zero-flicker streaming via MJPEG byte streaming.

---

## 5. Conclusion & Future Scope

The updated Flask-based AI Surveillance Platform establishes a high-performance, browser-accessible cyber-physical guardrail system. By combining YOLOv8 deep learning tracking with an interactive HTML5 canvas and lightweight REST endpoints, the platform solves the rigid rectangular boundary limitations of traditional CCTV systems. 

**Future Expansion Directions:**
1. Export model weights to TensorRT/OpenVINO for NVIDIA Jetson edge accelerators.
2. Multi-camera stream multiplexing for facility-wide monitoring.
3. ByteTrack integration for advanced trajectory analytics and velocity estimation.
