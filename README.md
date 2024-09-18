#  Face Detection Project

##  Overview

* This project implements real-time face detection using OpenCV and Haar Cascades.

##  Prerequisites

* Python 3.x
* OpenCV
* Haar Cascade XML file for face detection

##  Setup Instructions

1. * Clone the repository: `git clone https://github.com/Endekalu777/Face-detection_opencv.git`
2. * Navigate to the project directory: `cd Face_detection`
3. * Install dependencies: `pip install opencv-python`
4. * Make sure haarcascade_frontalface_default.xml is in the correct location

##  Running the Project

* Run the Python script to start detecting faces from webcam:

```bash
python app/face_detect.py
```

## Project Structure

* app/
* face_detect.py - Main script for face detection
* haarcascade_frontalface_default.xml - Haar cascade for face detection [click here](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)

## How It Works

* OpenCV captures video frames from the webcam
* The frames are converted to grayscale for processing
* Haar Cascade detects faces in the frame
* Detected faces are highlighted with a rectangle

## Author

* Endekalu Simon Haile