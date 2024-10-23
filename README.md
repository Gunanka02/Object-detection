# Object-detection
This project is designed to enhance the integrity of online quizzes or exams by detecting the use of mobile phones during the session. It leverages computer vision techniques, specifically YOLO (You Only Look Once), a pre-trained object detection model, to monitor the video feed in real-time and identify any phones visible in the camera's view.


# 1.Setting Up Your Environment
Install Python and set up virtual environment 

(Install virtualenv if you haven't already)
pip install virtualenv

(Create a new virtual environment)
virtualenv phone_detection_env

Activate the virtual environment
(On Windows)
phone_detection_env\Scripts\activate
(On macOS/Linux)
source phone_detection_env/bin/activate

Install Required Libraries

!pip install opencv-python numpy tensorflow keras

# 2.Download Necessary Files (YOLO Model)

YOLOv3 Weights (https://sourceforge.net/projects/yolov3.mirror/files/v8/yolov3.weights/download)

YOLOv3 Config (https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)

COCO dataset (https://github.com/pjreddie/darknet/blob/master/data/coco.names)

Place these files in your project directory.
 
# 3.Create a New Python Script

Create a file named phone_detection.py 

# 4.Run the Script

python phone_detection.py


# Object Detection
This code performs real-time detection of all objects visible in the webcam feed using the YOLO model trained on the COCO dataset. It identifies multiple object types simultaneously from a set of 80 classes, including people, animals, vehicles, and household items. The code analyzes each video frame as it is captured, drawing bounding boxes around detected objects and labeling them with their respective names and confidence scores. A confidence threshold of 0.5 is applied to filter out uncertain detections, ensuring efficient and effective object identification in any scene captured by the webcam.
