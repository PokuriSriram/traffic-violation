import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
from shapely.geometry import Point, Polygon, LineString

# --- CONFIGURATION ---
VIDEO_PATH = "traffic.mp4"  # Replace with your video path or 0 for webcam
MODEL_NAME = "yolov8n.pt"   # 'n' for speed, 'm'/'l' for accuracy
OUTPUT_PATH = "output_analysis.avi"

# Camera Calibration (Simulated)
# In production, use cv2.getPerspectiveTransform()
PIXELS_PER_METER = 20  # Example value: adjust based on camera height/angle
FPS = 30               # Frame rate of input video

# Analytics Thresholds
SPEED_LIMIT_KMPH = 60
RASH_ACCEL_THRESH = 3.0  # m/s^2 (Approx 10.8 kmph/s)
QUEUE_ROI_COORDS = [(200, 400), (1000, 400), (1100, 900), (100, 900)] # Example Trapezoid
STOP_LINE_COORDS = [(200, 600), (1000, 600)]

# Visuals
COLORS = {
    'car': (0, 255, 0),
    'bus': (0, 255, 255),
    'truck': (0, 165, 255),
    'motorcycle': (255, 255, 0),
    'violation': (0, 0, 255)
}

class TrafficAnalytics:
    def __init__(self):
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.speeds = {}
        self.violations = set() # Store track_ids of violators
        self.queue_density = 0.0
        
        # Define Zones
        self.queue_poly = Polygon(QUEUE_ROI_COORDS)
        self.stop_line = LineString(STOP_LINE_COORDS)
        self.red_light_active = True  # Simulated Signal State

    def _estimate_speed(self, track_id, current_center):
        """
        Estimates speed using Euclidean distance over time.
        Math: Speed = (Distance_Pixels / PPM) * FPS * 3.6 (to km/h)
        """
        if len(self.track_history[track_id]) < 2:
            return 0.0
            
        prev_center = self.track_history[track_id][-2]
        
        # Euclidean distance in pixels
        dist_px = np.linalg.norm(np.array(current_center) - np.array(prev_center))
        
        # Convert to real-world metrics
        dist_m = dist_px / PIXELS_PER_METER
        speed_mps = dist_m * FPS
        speed_kmph = speed_mps * 3.6
        
        # Exponential Moving Average for smoothing
        prev_speed = self.speeds.get(track_id, {}).get('speed', speed_kmph)
        smoothed_speed = 0.8 * speed_kmph + 0.2 * prev_speed

        # Calculate Acceleration (Delta Speed / Delta Time)
        # Delta Time is 1/FPS
        acceleration = (smoothed_speed - prev_speed) / (1 / FPS) # km/h/s
        
        return {'speed': smoothed_speed, 'acceleration': acceleration}

    def _detect_violations(self, track_id, current_center, speed_data):
        """
        Checks for:
        1. Red Light Jumping (Line Intersection)
        2. Over-speeding
        3. Rash Driving (High Acceleration/Deceleration)
        """
        violation_types = []
        speed = speed_data['speed']
        accel = speed_data['acceleration']
        
        # 1. Red Light Violation
        # We check if the movement vector intersects the stop line
        if self.red_light_active and len(self.track_history[track_id]) >= 2:
            prev_center = self.track_history[track_id][-2]
            movement_vector = LineString([prev_center, current_center])
            
            if movement_vector.intersects(self.stop_line):
                violation_types.append("Red Light Jump")
        
        # 2. Speed Violation
        if speed > SPEED_LIMIT_KMPH:
            violation_types.append("Overspeeding")

        # 3. Rash Driving (Acceleration)
        # RASH_ACCEL_THRESH is in m/s^2 approx, but let's use a raw value for km/h/s
        # 10 km/h/s is approx 2.7 m/s^2, which is hard braking/acceleration
        if abs(accel) > 15: # Threshold of 15 km/h per second change
            violation_types.append("Rash Driving")
            
        return violation_types

    def update(self, detections, frame_shape):
        """
        Main update loop for analytics.
        detections: List of [x1, y1, x2, y2, track_id, class_id]
        """
        current_queue_count = 0
        
        for *box, track_id, class_id in detections:
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Update History
            self.track_history[track_id].append(center)
            
            # --- Analytics ---
            
            # 1. Speed & Acceleration Estimation
            speed_data = self._estimate_speed(track_id, center)
            self.speeds[track_id] = speed_data
            
            # 2. Violation Check
            new_violations = self._detect_violations(track_id, center, speed_data)
            if new_violations:
                self.violations.add(track_id)
            
            # 3. Queue Calculation
            # Check if vehicle centroid is inside ROI polygon
            if self.queue_poly.contains(Point(center)):
                current_queue_count += 1
                
        # Calculate Density (Vehicles per 1000 sq pixels for simplicity, or just count)
        # Real density = Count / (Area_Meters)
        self.queue_density = current_queue_count # Simple count for dashboard

def draw_hud(frame, analytics_engine):
    """
    Draws the "Iron Man" style interface.
    """
    # Draw ROI
    cv2.polylines(frame, [np.array(QUEUE_ROI_COORDS, np.int32)], True, (200, 200, 200), 2)
    cv2.putText(frame, "QUEUE ZONE", (QUEUE_ROI_COORDS[0][0], QUEUE_ROI_COORDS[0][1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Draw Stop Line
    color = (0, 0, 255) if analytics_engine.red_light_active else (0, 255, 0)
    cv2.line(frame, STOP_LINE_COORDS[0], STOP_LINE_COORDS[1], color, 3)
    cv2.putText(frame, f"SIGNAL: {'RED' if analytics_engine.red_light_active else 'GREEN'}", 
                (STOP_LINE_COORDS[0][0], STOP_LINE_COORDS[0][1]+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Dashboard Metrics
    cv2.rectangle(frame, (20, 20), (300, 150), (0, 0, 0), -1)
    cv2.putText(frame, f"Queue Count: {analytics_engine.queue_density}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Violations: {len(analytics_engine.violations)}", (30, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame

def main():
    print(f"Loading Model: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    
    # Initialize Engine
    analytics = TrafficAnalytics()
    
    # Video Setup
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}. Using webcam or check path.")
        # cap = cv2.VideoCapture(0) # Uncomment for webcam
        return

    # Output Setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'XVID'), FPS, (width, height))

    print("Starting Inference Pipeline...")
    print("Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Detection & Tracking (Ultralytics built-in ByteTrack)
        # persist=True is CRITICAL for tracking to maintain IDs across frames
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        if results[0].boxes.id is not None:
            # Extract boxes: x1, y1, x2, y2, id, conf, class
            boxes = results[0].boxes.data.cpu().numpy()
            
            # 2. Update Analytics
            analytics.update(boxes, frame.shape)
            
            # 3. Annotate Frame
            for box in boxes:
                x1, y1, x2, y2, track_id, conf, cls_id = box
                track_id = int(track_id)
                speed_data = analytics.speeds.get(track_id, {'speed': 0, 'acceleration': 0})
                speed = speed_data['speed']
                accel = speed_data['acceleration']
                
                # Check if violator
                is_violator = track_id in analytics.violations
                color = COLORS['violation'] if is_violator else COLORS.get('car', (0, 255, 0))
                
                # Bounding Box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Label: ID | Speed
                label = f"ID:{track_id} | {int(speed)}km/h"
                if abs(accel) > 15:
                    label += " [RASH]"
                if is_violator:
                    label += " [VIOLATION]"
                
                cv2.putText(frame, label, (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 4. Draw HUD
        frame = draw_hud(frame, analytics)
        
        # 5. Interface Control
        # Toggle Red Light simulation with 's'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'): 
            analytics.red_light_active = not analytics.red_light_active
            
        out.write(frame)
        cv2.imshow("Traffic Analytics System", frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing Complete. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
