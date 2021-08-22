# WinkDetectionOpenCV.py
#
# Description:
#   Contains a function that counts the number of winks in a frame. This
#   function uses OpenCV's haar cascades for facial and eye detections. The
#   function takes three inputs:
#       - An image or video frame (frame)
#       - OpenCV's face haar cascade (faceCascade)
#       - OpenCV's eye haar cascade (eyesCascade)
#
#   This function uses OpenCV's detectMultiScale for both facial and eye
#   detection and defines a wink as a single eye detection.
#
#   There is some image preprocessing done on the image:
#       - Before face detection:
#           - Convert to grayscale
#           - Gaussian blurring
#           - Histogram equalization
#       - Before eye detection:
#           - Cropping
#           - Convert to grayscale
#           - Median blurring
#
#   All faces detected are surrounded by a colored rectangle. The color of the
#   rectangle indicates whether the function detects a wink or not:
#       - Blue rectangle:  Winking
#       - Green rectangle: Not winking
#   All single eye detections are also surrounded by a red colored rectangle.


import cv2
import copy


# Detects faces and uses detectWink function to detect wink
#   Returns number of winks detected
def detectFaceAndWink(frame, faceCascade, eyesCascade):
    origFrame = copy.deepcopy(frame)
    
    # Preprocessing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (3,3), 0)
    gray_frame = cv2.equalizeHist(gray_frame)
	
    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray_frame,					# image
		1.12, 						# scaleFactor
		4, 							# minNeighbors
		0|cv2.CASCADE_SCALE_IMAGE,	# flag
		(10, 10))					# minSize

    # Loop through detected faces
    detected = 0
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        
        # Cropping
        faceROI = origFrame[y:y+h, x:x+w]
        if detectWink(frame, (x, y), faceROI, eyesCascade):
            detected += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2) # Blue Rect = wink
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2) # Green Rect = not a wink
    return detected


# Function to detect winking
#   Returns true if face is winking and false otherwise
def detectWink(frame, location, ROI, eyesCascade):
    # Preprocessing
    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    ROI = ROI[0:int(2.0*ROI.shape[0]/3.0), :]
    ROI = cv2.medianBlur(ROI, 3)
	
    # Eye detection
    eyes = eyesCascade.detectMultiScale(
        ROI,						# image
		1.13, 						# scaleFactor
		5, 							# minNeighbors
		0|cv2.CASCADE_SCALE_IMAGE,	# flag
		(0, 0))						# minSize
	
    # Loop through detected eyes
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
		
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2) # Red Rect = eye
    return len(eyes) == 1    # number of eyes is one