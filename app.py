import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO
from collections import defaultdict, deque
from shapely.geometry import Point, Polygon, LineString

# --- PAGE CONFIG ---
# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Team Spy | Traffic Analytics",
    page_icon="�",
    layout="wide"
)

# --- SIDEBAR CONFIG ---
st.sidebar.title("🔧 Configuration")
model_confidence = st.sidebar.slider("Model Confidence", 0.0, 1.0, 0.3)
speed_smoothing = st.sidebar.slider("Speed Smoothing Factor", 0.0, 1.0, 0.8)
enable_red_light = st.sidebar.checkbox("Enable Red Light Simulation", value=True)

# --- CONSTANTS & CONFIG ---
MODEL_NAME = "yolov8n.pt"
# Camera Calibration (Simulated)
PIXELS_PER_METER = 20  
SPEED_LIMIT_KMPH = 60
QUEUE_ROI_COORDS = [(200, 400), (1000, 400), (1100, 900), (100, 900)] 
STOP_LINE_COORDS = [(200, 600), (1000, 600)]

COLORS = {
    'car': (0, 255, 0),
    'bus': (0, 255, 255),
    'truck': (0, 165, 255),
    'motorcycle': (255, 255, 0),
    'violation': (0, 0, 255)
}

# --- ANALYTICS ENGINE (Adapted) ---
# --- ANALYTICS ENGINE (Enhanced with Trajectory Logic) ---
class TrafficAnalytics:
    def __init__(self):
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.speeds = {}
        self.violations = set() 
        self.violation_log = []
        self.queue_density = 0.0
        self.queue_ids = set()
        self.vehicle_counts = defaultdict(int)
        self.counted_ids = set()
        
        # Zones
        self.queue_poly = Polygon(QUEUE_ROI_COORDS)
        self.stop_line = LineString(STOP_LINE_COORDS)
        self.red_light_active = True 

    def _get_speed_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _check_rash_driving_trajectory(self, track_id, fps):
        """
        User's requested trajectory-based logic:
        Iterate through recent history to find sudden acceleration spikes.
        """
        traj = list(self.track_history[track_id])
        if len(traj) < 4:
            return False

        # Calculate speeds between consecutive points in the trajectory
        raw_distances = []
        for i in range(1, len(traj)):
            dist = self._get_speed_distance(traj[i-1], traj[i])
            raw_distances.append(dist)
            
        # Check acceleration (difference in distance/speed)
        # We look for a jump greater than ACCEL_THRESHOLD in consecutive segments
        # Adjust threshold relative to FPS to match "per frame" logic or physical units
        # User used ACCEL_THRESHOLD=15 (likely raw pixel diff or similar). 
        # We will use the same heuristic logic but scaled.
        
        for i in range(1, len(raw_distances)):
            # Delta Distance (Proxy for acceleration)
            delta = abs(raw_distances[i] - raw_distances[i-1])
            
            # Heuristic: 15 pixels jump per frame diff (~ high jerk)
            # Scaling by 30/FPS to normalize if video slows down
            if delta > (15 * (30/fps)): 
                return True
                
        return False

    def _estimate_live_speed(self, track_id, current_center, fps):
        if len(self.track_history[track_id]) < 2:
            return 0.0
            
        prev_center = self.track_history[track_id][-2]
        dist_px = self._get_speed_distance(current_center, prev_center)
        dist_m = dist_px / PIXELS_PER_METER
        speed_kmph = (dist_m * fps) * 3.6
        
        # Smoothing
        prev_speed = self.speeds.get(track_id, speed_kmph)
        smoothed_speed = speed_smoothing * speed_kmph + (1 - speed_smoothing) * prev_speed
        return smoothed_speed

    def update(self, detections, fps):
        self.queue_ids = set() # Reset for current frame
        
        for det in detections:
            # Safe unpacking
            if len(det) == 7:
                x1, y1, x2, y2, track_id, conf, class_id = det
            else:
                continue
                
            track_id = int(track_id)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # 1. Update Trajectory
            self.track_history[track_id].append(center)
            
            # 2. Update Counts (Once per ID)
            if track_id not in self.counted_ids:
                self.counted_ids.add(track_id)
                self.vehicle_counts['total'] += 1
            
            # 3. Speed Calculation
            speed = self._estimate_live_speed(track_id, center, fps)
            self.speeds[track_id] = speed
            
            # 4. Rash Driving Check (Trajectory Logic)
            if self._check_rash_driving_trajectory(track_id, fps):
                if track_id not in self.violations:
                    self.violations.add(track_id)
                    self.violation_log.append({
                        "ID": track_id, "Type": "Rash Driving", 
                        "Speed": f"{int(speed)} km/h", "Time": time.strftime("%H:%M:%S")
                    })

            # 5. Red Light Violation
            if self.red_light_active and len(self.track_history[track_id]) >= 2:
                prev_center = self.track_history[track_id][-2]
                movement_vector = LineString([prev_center, center])
                if movement_vector.intersects(self.stop_line):
                    if track_id not in self.violations:
                        self.violations.add(track_id)
                        self.violation_log.append({
                            "ID": track_id, "Type": "Red Light Jump", 
                            "Speed": f"{int(speed)} km/h", "Time": time.strftime("%H:%M:%S")
                        })
            
            # 6. Queue Logic
            if self.queue_poly.contains(Point(center)):
                self.queue_ids.add(track_id)
                self.queue_density = len(self.queue_ids)

# --- HELPERS ---
def draw_overlay(frame, analytics):
    # 1. Draw Queue Poly
    cv2.polylines(frame, [np.array(QUEUE_ROI_COORDS, np.int32)], True, (200, 200, 200), 2)
    
    # 2. Draw Stop Line
    color = (0, 0, 255) if analytics.red_light_active else (0, 255, 0)
    cv2.line(frame, STOP_LINE_COORDS[0], STOP_LINE_COORDS[1], color, 3)
    
    # 3. Draw Trajectories (New Feature)
    for track_id, points in analytics.track_history.items():
        if len(points) > 2:
            pts = np.array(list(points), np.int32)
            pts = pts.reshape((-1, 1, 2))
            is_violator = track_id in analytics.violations
            traj_color = (0, 0, 255) if is_violator else (255, 255, 0)
            cv2.polylines(frame, [pts], False, traj_color, 2)

    return frame

# --- MAIN APP ---
def main():
    st.title("🚦 Team Spy Traffic Analytics")
    st.markdown("Run advanced traffic analysis on any video file.")

    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload Traffic Video (MP4/AVI)", type=['mp4', 'avi', 'mov'])

    if uploaded_file is not None:
        # Save to temp file because OpenCV needs a path
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Load Model
        with st.spinner("Loading YOLOv8 Model..."):
            model = YOLO(MODEL_NAME)
        
        analytics = TrafficAnalytics()
        analytics.red_light_active = enable_red_light
        
        # Dashboard Layout
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Live Feed")
            image_placeholder = st.empty()
        
        with col2:
            st.markdown("### Real-Time Metrics")
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st_queue = st.metric("Queue Count", 0)
            with kpi2:
                st_violation = st.metric("Total Violations", 0)
            with kpi3:
                st_count = st.metric("Total Vehicles", 0)
                
            st.markdown("### Violation Log")
            log_placeholder = st.empty()

        # Processing Loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run Inference
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", 
                                  conf=model_confidence, verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.data.cpu().numpy()
                analytics.update(boxes, fps)
                analytics.red_light_active = enable_red_light # Sync with UI
                
                # Annotate
                for box in boxes:
                    x1, y1, x2, y2, track_id, conf, cls_id = box
                    track_id = int(track_id)
                    speed = analytics.speeds.get(track_id, 0) # Now just a float
                    
                    is_violator = track_id in analytics.violations
                    is_in_queue = track_id in analytics.queue_ids
                    
                    if is_violator:
                        color = (255, 0, 0)
                    elif is_in_queue:
                        color = (0, 165, 255) # Orange for queue
                    else:
                        color = (0, 255, 0)
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"ID:{track_id} {int(speed)}km"
                    cv2.putText(frame, label, (int(x1), int(y1)-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw Overlay
            frame = draw_overlay(frame, analytics)

            # Update UI
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update Metrics
            st_queue.metric("Queue Count", analytics.queue_density)
            st_violation.metric("Total Violations", len(analytics.violations))
            st_count.metric("Total Vehicles", analytics.vehicle_counts['total'])
            
            if analytics.violation_log:
                df = pd.DataFrame(analytics.violation_log)
                log_placeholder.dataframe(df.tail(5), use_container_width=True, hide_index=True)
            else:
                log_placeholder.info("No violations detected yet.")
                
        cap.release()
        st.success("Analysis Complete!")

if __name__ == "__main__":
    main()
