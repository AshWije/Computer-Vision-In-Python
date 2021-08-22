# Facial and Eye Detection

This folder contains the implementation of facial and eye detections with OpenCV and MTCNNs
in order to perform wink detection and for an interactive program to place glasses on the
faces of people.

Two of the files perform wink detection, one with an MTCNN for facial detection and one
with OpenCV's haar cascades for facial detection. The eye detection also varies from OpenCV's
haar cascade to a multi-stage detector. This was done to compare performance of the two
variations.

A brief description of each file is shown below:


**Place Glasses OpenCV**
Contains a function that places glasses over the eyes of all faces in a
frame and returns the total number of winks that it detects. This program
is similar to WinkDetectionOpenCV.py for facial and eye detections in using
OpenCV's haar cascades for them.


**Wink Detection MTCNN**
Contains a function that determines whether a person is winking. This
function requires MTCNN (from mtcnn_cv2 import MTCNN) for the facial
detection.


**Wink Detection OpenCV**
Contains a function that counts the number of winks in a frame. This
function uses OpenCV's haar cascades for facial and eye detections.


2019, 2021