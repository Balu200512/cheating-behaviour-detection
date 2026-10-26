# Cheating Behaviour Detection Using Head Movement

## Project Overview
This project detects potentially suspicious behaviour during examinations by monitoring head movements using YOLOv8 and a webcam. It identifies head directions and triggers an audio alert when a suspicious direction is detected continuously for more than 5 seconds.

## Features
- Real-time webcam monitoring
- Head movement classification
- Detection of left and right head movements as potentially suspicious
- Audio alert after 5 seconds of continuous suspicious movement
- Live bounding boxes and confidence scores

## Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Computer Vision

## Project Files
- `train.py` – Trains the YOLOv8 model using the configured dataset.
- `run_webcam.py` – Runs real-time webcam detection and alert logic.
- `data.yaml` – Defines dataset paths and class names.

## Requirements
- Python 3.10 or compatible version
- OpenCV
- Ultralytics
- A working webcam

## Installation

Install the required libraries:

```bash
pip install ultralytics opencv-python
```

## Dataset Configuration
Configure `data.yaml` with the correct paths to your training and validation images and labels.

## Training

```bash
python train.py
```

## Run Detection

Update the model path in `run_webcam.py` to point to your trained `best.pt` weights, then run:

```bash
python run_webcam.py
```

Press `q` to close the webcam window.

## Disclaimer
This project is an academic prototype. Head movement alone does not prove cheating, and detections may produce false positives. Use it as an assistive tool, not as the sole basis for disciplinary decisions.

## Author
Balu200512
