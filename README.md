📛 Traffic Violation Detection & Queue Estimation

A Python‑based system to detect traffic violations (e.g., red‑line crossing and rash driving by analyzing vehicle trajectories) and estimate queue length/size using mathematical formulas — bundled with a Streamlit web app for visualization and interactive inference.

🧠 Overview

This project implements:

🚦 Red‑line violation detection — identifies vehicles crossing a predefined stop‑line at traffic signals.

🚗 Rash driving detection — analyzes vehicle trajectories to identify erratic (rashy) movement patterns.

📊 Queue length & size estimation — estimates how many vehicles are waiting in a lane (and approximate queue length) via mathematical modeling.

🖥️ Streamlit app — user‑friendly interface to upload video feeds and visualize results interactively.

📂 Repository Structure
traffic‑violation/
├── app.py                 # Streamlit web application
├── main.py                # Main processing logic for detection & estimation
├── requirements.txt       # Python dependencies
├── yolov8n.pt             # YOLOv8 object detection model
├── README.md              # This file (usage guide)
└── DESIGN_DOC.md          # Design details and technical explanations

🚀 Features
✔️ Traffic Violations Covered
Feature	Description
Red‑line Violation	Detect vehicles crossing the stop line during a red signal.
Rash Driving Detection	Use object tracking & trajectory analysis to flag dangerous/erratic driving.
Queue Estimation	Compute the number of vehicles and approximate queue length using geometry & math.
🛠️ Getting Started
📌 Prerequisites

Install Python 3.8+ and the packages in requirements.txt:

python3 -m venv venv
source venv/bin/activate         # macOS/Linux 
venv\Scripts\activate            # Windows
pip install -r requirements.txt

🧪 Running the Streamlit App

The Streamlit app provides an interactive UI for video upload and viewing results:

streamlit run app.py


This starts the UI in your browser — upload a video and view:

✔ Red‑line violation overlays
✔ Rash driving flags
✔ Queue length/size estimates

🧠 How It Works (High‑Level)

Object Detection with YOLOv8

Detect vehicles frame‑by‑frame.

Output bounding boxes for tracking.

Vehicle Tracking & Trajectory Analysis

Track detected vehicles across video frames.

Compute trajectory curvature/speed to identify rash behavior.

Violation Rules

If a vehicle crosses a stop line during red traffic phase → red‑line violation.

If trajectory behavior exceeds thresholds → rash driving flag.

Queue Estimation

Use detected vehicles near the signal to estimate queue size (count) and length (meters) using simple geometric formulas and bounding box positions.

📦 Example Use Cases

✔ Traffic monitoring at intersections
✔ Dashboard for traffic enforcement officers
✔ Intelligent transport systems
✔ Research and smart city applications

🧪 Sample Output

When you run on a traffic video, the app:

Displays the video frames with bounding boxes

Marks violations (colored overlays)

Shows queue estimation stats on sidebar
