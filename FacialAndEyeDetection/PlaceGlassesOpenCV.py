# PlaceGlassesOpenCV.py
#
# Description:
#   Contains a function that places glasses over the eyes of all faces in a
#   frame and returns the total number of winks that it detects. This program
#   is similar to WinkDetectionOpenCV.py for facial and eye detections in using
#   OpenCV's haar cascades for them. The function takes four inputs:
#       - An image or video frame (frame)
#       - OpenCV's face haar cascade (faceCascade)
#       - OpenCV's eye haar cascade (eyesCascade)
#       - The glasses png image (glassesImage)
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
#   Different actions are performed in placing the glasses for different numbers
#   of eye detections. These differences are shown by the different functions
#   defined for each case (two eyes, left eye, and right eye).


import numpy as np
import cv2
import copy


# Detects faces and uses detectWink function to detect wink
#   Returns number of winks detected
def detectFaceAndWink(frame, faceCascade, eyesCascade, glassesImage):
    origFrame = copy.deepcopy(frame)

    # Preprocessing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (3,3), 0)
    gray_frame = cv2.equalizeHist(gray_frame)
	
    # Face detection
    faces = faceCascade.detectMultiScale(
        gray_frame,				    # image
		1.12, 						# scaleFactor
		4, 							# minNeighbors
		0|cv2.CASCADE_SCALE_IMAGE,	# flag
		(10, 10))					# minSize

    # Loop through detected faces
    detected = 0
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        faceROI = origFrame[y:y+h, x:x+w]
        if detectWink(frame, (x, y), faceROI, eyesCascade, glassesImage):
            detected += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2) # Blue Rect = wink
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2) # Green Rect = not a wink
    return detected

# Function to detect winking
#   Returns true if face is winking and false otherwise
def detectWink(frame, location, ROI, eyesCascade, glassesImage):
    # Preprocessing
    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    ROI = ROI[0:int(2.0*ROI.shape[0]/3.0), :]
    ROI = cv2.equalizeHist(ROI)
	
    # Eye detection
    eyes = eyesCascade.detectMultiScale(
        ROI,						# image
		1.15, 						# scaleFactor
		5, 							# minNeighbors
		0|cv2.CASCADE_SCALE_IMAGE,	# flag
		(0, 0))						# minSize
	
    # Loop through detected eyes
    leftmostEye = None
    rightmostEye = None
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]

        if leftmostEye == None or leftmostEye[0] > x:
            leftmostEye = (x, y, w, h)
        if rightmostEye == None or rightmostEye[0] < x:
            rightmostEye = (x, y, w, h)
		
	# if there are 2 eyes
    if len(eyes) == 2: glassesTwoEyes(leftmostEye, rightmostEye, frame, glassesImage)
		
	# if there is only 1 eye
    elif len(eyes) == 1:
		# determine if eye is on the left or right
        if leftmostEye[0] - location[0] + (leftmostEye[2]/2.0) <= ROI.shape[1] / 2.0: glassesLeftEye(leftmostEye, frame, location, glassesImage)
        else: glassesRightEye(leftmostEye, frame, location, glassesImage)

    return len(eyes) == 1    # number of eyes is one


# Place glasses on person's face at given location
def placeGlasses(y1, y2, x1, x2, glasses, frame):
    if y1 >= 0 and x1 >= 0 and y2 <= frame.shape[0] and x2 <= frame.shape[1]:
        alpha = glasses[:, :, 3] / 255.0
	
        alphaGlasses = np.zeros((glasses.shape[0], glasses.shape[1], 3), dtype=np.uint8)
        alphaGlasses[:, :, 0] = (1.0 - alpha) * frame[y1:y2, x1:x2, 0] + alpha * glasses[:, :, 0]
        alphaGlasses[:, :, 1] = (1.0 - alpha) * frame[y1:y2, x1:x2, 1] + alpha * glasses[:, :, 1]
        alphaGlasses[:, :, 2] = (1.0 - alpha) * frame[y1:y2, x1:x2, 2] + alpha * glasses[:, :, 2]
	
        frame[y1:y2, x1:x2] = alphaGlasses

# Determines location to place glasses when both eyes are visible
def glassesTwoEyes(leftEye, rightEye, frame, glassesImage):
    glasses = cv2.imread(glassesImage, cv2.IMREAD_UNCHANGED)
    width = 1.0 * (rightEye[0]-leftEye[0]+leftEye[2])
    glasses = cv2.resize(glasses, (int(width), int(glasses.shape[0] * (width) / glasses.shape[1])))
	
    y1, y2 = leftEye[1], leftEye[1]+glasses.shape[0]
    x1, x2 = leftEye[0], leftEye[0]+glasses.shape[1]
	
    placeGlasses(y1, y2, x1, x2, glasses, frame)

# Determines location to place glasses when only the left eye is visible
def glassesLeftEye(eye, frame, location, glassesImage):
    glasses = cv2.imread(glassesImage, cv2.IMREAD_UNCHANGED)
    width = 2.0 * (eye[0]-location[0]+eye[2])
    glasses = cv2.resize(glasses, (int(width), int(glasses.shape[0] * (width) / glasses.shape[1])))
	
    y1, y2 = eye[1], eye[1]+glasses.shape[0]
    x1, x2 = eye[0], eye[0]+glasses.shape[1]
	
    placeGlasses(y1, y2, x1, x2, glasses, frame)

# Determines location to place glasses when only the right eye is visible
def glassesRightEye(eye, frame, location, glassesImage):
    glasses = cv2.imread(glassesImage, cv2.IMREAD_UNCHANGED)
    width = 1.0 * (eye[0]-location[0]+eye[2])
    glasses = cv2.resize(glasses, (int(width), int(glasses.shape[0] * (width) / glasses.shape[1])))
	
    y1, y2 = eye[1], eye[1]+glasses.shape[0]
    x1, x2 = eye[0]-(2*eye[2]), eye[0]-(2*eye[2])+glasses.shape[1]
	
    placeGlasses(y1, y2, x1, x2, glasses, frame)