# AI-Powered Traffic Analysis & Violation Detection System

## 1. Project Overview
This project implements a modular, explainable computer vision system for traffic video analysis. It focuses on vehicle detection, multi-object tracking, queue analytics, and violation detection (red light, rash driving) specifically tailored for Indian road conditions.

## 2. Engineering Approach & Validations
### Why YOLOv8 + ByteTrack?
- **YOLOv8**: State-of-the-art accuracy/speed trade-off. It handles varying distinct classes (Car, Auto-rickshaw, Bus, Bike) well, which is crucial for Indian mixed traffic.
- **ByteTrack**: A "Tracking-by-Detection" algorithm. Unlike DeepSORT, it uses low-confidence detection boxes (which usually represent occluded objects) to maintain tracklets, reducing identification switches in dense traffic queues.

### Explainability
- **Speed Est**: Uses pixel-to-meter calibration (PPM). $Speed = \frac{Distance_{pixels} \times PPM}{Time_{seconds}}$.
- **Queue Density**: Calculated as the ratio of vehicle area vs. Total ROI area.
- **Rash Driving**: Derived from trajectory variance and instantaneous acceleration thresholds.

## 3. Training Phase (Step-by-Step)
### 3.1 Datasets
For Indian context, standard COCO is "okay" but misses Auto-rickshaws.
- **Recommended**: **IDD (Indian Driving Dataset)** by IIIT Hyderabad.
- **Alternative**: Mix COCO vehicles + custom scraped specific classes.

### 3.2 Annotation Strategy
- **Tool**: CVAT (Computer Vision Annotation Tool).
- **Labeling**: Tight bounding boxes. Occlusion: label visible part or estimate full extent if clear.
- **Why Frames?**: We train on randomized static frames to learn object features invariant of time. Tracking temporal info is handled by ByteTrack, not the CNN.

### 3.3 Folder Structure for Training
```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

### 3.4 Colab Training Snippet
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # build a new model from scratch

# Train the model
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='traffic_india_v1'
)
```

## 4. Analytics Logic (The "Math")

### 4.1 Queue Analytics
- **ROI**: A Polygon defined at the lane level.
- **Logic**: Count unique Track IDs with centroids inside the ROI Polygon.
- **Density**: $\frac{\sum(VehicleBoxArea)}{ROI\_Area}$.

### 4.2 Violation Detection
#### Red Light Violation (RLVD)
1. **Inputs**: Signal State (External or Detected), Stop Line (Line Segment).
2. **Logic**:
   - If Signal == Red:
     - Monitor Vehicle Line Intersection.
     - Vectors: $A, B$ (Stop line points), $C, D$ (Vehicle movement vector).
     - If $CD$ intersects $A, B$ $\rightarrow$ Violation.

#### Rash Driving
1. **Speed**: Track centroid $P(t)$ vs $P(t-1)$. Calculate Euclidean distance.
2. **Acceleration**: $a = \frac{v_t - v_{t-1}}{\Delta t}$.
   - If $a > Threshold_{high}$: Sudden Start.
   - If $a < Threshold_{low}$: Hard Braking.
3. **Zig-Zag**: Compute lateral deviation from the lane center curve over a temporal window.

## 5. Evaluation & Future Improvements
- **Metrics**: MOTA (Multi-Object Tracking Accuracy), IDF1 (ID F1 Score).
- **Future**:
  - **Transformer Tracking**: Use TrackFormer for end-to-end learnable tracking logic.
  - **ANPR**: Integrate OCR (PaddleOCR) on violation cropped frames.
