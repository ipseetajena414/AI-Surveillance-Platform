import cv2
import argparse
import time
from surveillance import SurveillanceSystem

# Global variables for ROI drawing
drawing = False
roi_polygon = []

def draw_roi(event, x, y, flags, param):
    global drawing, roi_polygon
    frame_copy = param['frame'].copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_polygon.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    # Draw the current polygon
    if len(roi_polygon) > 0:
        for i in range(len(roi_polygon) - 1):
            cv2.line(frame_copy, roi_polygon[i], roi_polygon[i+1], (0, 255, 0), 2)
        if not drawing and len(roi_polygon) > 2:
            # Close the polygon
            cv2.line(frame_copy, roi_polygon[-1], roi_polygon[0], (0, 255, 0), 2)
            
        for pt in roi_polygon:
            cv2.circle(frame_copy, pt, 4, (0, 0, 255), -1)

    cv2.imshow('Define ROI (Press ENTER to finish, c to clear)', frame_copy)


def main():
    parser = argparse.ArgumentParser(description="AI Surveillance Analytics System")
    parser.add_argument('--source', type=str, default='0', help='Video source: 0 for webcam, or path to video file')
    parser.add_argument('--crowd_threshold', type=int, default=3, help='Number of people to trigger crowd alert')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLOv8 model')
    parser.add_argument('--save', action='store_true', help='Save output video to output.mp4')
    args = parser.parse_args()

    # Initialize video capture
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return

    # Read first frame to define ROI
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from video source")
        return

    print("\n--- Setup ROI ---")
    print("Click to add points to the ROI polygon.")
    print("Press 'ENTER' or 'SPACE' when finished.")
    print("Press 'c' to clear the current polygon.")
    print("-----------------\n")

    param = {'frame': frame}
    cv2.namedWindow('Define ROI (Press ENTER to finish, c to clear)')
    cv2.setMouseCallback('Define ROI (Press ENTER to finish, c to clear)', draw_roi, param)
    cv2.imshow('Define ROI (Press ENTER to finish, c to clear)', frame)

    global roi_polygon
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 32:  # Enter or Space
            break
        elif key == ord('c'):
            roi_polygon.clear()
            cv2.imshow('Define ROI (Press ENTER to finish, c to clear)', frame)

    cv2.destroyWindow('Define ROI (Press ENTER to finish, c to clear)')
    
    print(f"ROI defined with {len(roi_polygon)} points: {roi_polygon}")

    # Initialize Surveillance System
    surveillance = SurveillanceSystem(model_path=args.model, crowd_threshold=args.crowd_threshold)

    # Setup Video Writer if requested
    out = None
    if args.save:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Handle cases where fps might be 0 (e.g. some webcams)
        fps = fps if fps > 0 else 30.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    print("Starting Surveillance... Press 'q' to quit.")

    # FPS Calculation variables
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error.")
            break

        # Process frame
        annotated_frame, _, _ = surveillance.process_frame(frame, roi_polygon=roi_polygon)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Draw FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (annotated_frame.shape[1] - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show Output
        cv2.imshow('AI Surveillance System', annotated_frame)

        # Save Output
        if out is not None:
            out.write(annotated_frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
