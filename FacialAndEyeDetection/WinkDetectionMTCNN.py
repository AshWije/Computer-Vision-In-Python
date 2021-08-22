# WinkDetectionMTCNN.py
#
# Description:
#   Contains a function that determines whether a person is winking. This
#   function requires MTCNN (from mtcnn_cv2 import MTCNN) for the facial
#   detection. The function takes three inputs:
#       - An image or video frame (frame)
#       - MTCNN face detection (f)
#       - OpenCV's eye haar cascade (eyesCascade)
#
#   This function has six stages in total:
#       1. Check face orientation
#       2. Calculate norm between colored eye pixels of both eyes
#       3. Use detectMultiScale and the difference of pixels
#       4. Use detectMultiScale and the difference of pixels
#       5. Use detectMultiScale
#       6. Use detectMultiScale
#
#   There is also a variety of image preprocessing done on the image:
#       - After stage 2:
#           - Convert to grayscale
#           - Crop above nose
#           - Median blurring
#       - After stage 5:
#           - Histogram equalization
#
#   The conditions for a face being labeled as winking or not varies for each
#   stage. These conditions are shown in the code below.


import cv2
import numpy as np


# Function to detect winking (more detail in top comment)
#   Returns true if face is winking and false otherwise
def wink(frame, f, eyesCascade) :
    rows, cols, bands = frame.shape
    box = f['box']
    x, y, w, h = box
    
    confidence = f['confidence']
    keypoints = f['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    nose = keypoints['nose']
    mouth_left = keypoints['mouth_left']
    mouth_right = keypoints['mouth_right']
    
    #""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # First stage -> Make sure that face is not sideways (one eye visible)
    if (nose[0] < left_eye[0] and nose[0] < mouth_left[0]) or (nose[0] > right_eye[0] and nose[0] > mouth_right[0]):
        return False
    
    #""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Second stage -> Use difference between eye pixels in color image
    
    # Get left eye pixel
    left_x, left_y = left_eye
    left_pixel = np.array(frame[left_y, left_x]).astype(np.float64)

    # Get right eye pixel
    right_x, right_y = right_eye
    right_pixel = np.array(frame[right_y, right_x]).astype(np.float64)

    # Get norm of difference between pixels
    norm_diff = np.linalg.norm(right_pixel - left_pixel)
    
    # If norm between eye pixels is less than 13, then the face is not winking
    if norm_diff < 13:
        return False
    
    #""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Third stage -> Use detect multi scale and the difference of pixels
    
    # Make image grayscale
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Crop the image to only the region of the face above the nose
    eyesFrame = grayFrame[y:nose[1], x:x+w]
    
    # Check to ensure that the face is not too small
    if eyesFrame.shape[0] == 0 and eyesFrame.shape[1] == 0:
        return False
    
    # Median blur the image
    eyesFrame = cv2.medianBlur(eyesFrame, 3)
    
    # Cascade classifier
    eyes = eyesCascade.detectMultiScale(
        eyesFrame,
        1.13,                       # scaleFactor
        5,                          # minNeighbors
        0|cv2.CASCADE_SCALE_IMAGE,  # flag
        (0,0))                      # minSize
   
    # If two eyes are detected, check norm of difference between pixels of eyes
    if (len(eyes) == 2):
    
        # Get detected values from eyes
        x1, y1, w1, h1 = eyes[0]
        x2, y2, w2, h2 = eyes[1]
        
        # Get max width and height between two eyes
        newW = max([w1, w2])
        newH = max([h1, h2])
        
        # Get eyes
        eye1Range = eyesFrame[y1:y1+h1, x1:x1+w1]
        eye2Range = eyesFrame[y2:y2+h2, x2:x2+w2]
        
        # Resize eyes to be the same size
        eye1Resize = cv2.resize(eye1Range, (newH, newW), interpolation = cv2.INTER_AREA)
        eye2Resize = cv2.resize(eye2Range, (newH, newW), interpolation = cv2.INTER_AREA)
        
        # Calculate norm of difference
        norm_diff = np.linalg.norm(eye1Resize - eye2Resize) / (newW * newH)
    
        # If norm between eye pixels is less than 2, then the face is not winking
        if norm_diff <= 2:
            return False
   
    # If only one eye is detected, then the face is winking
    elif len(eyes) == 1:
        return True
    
    #""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Fourth stage -> Use detect multi scale
    
    # Cascade classifier
    eyes = eyesCascade.detectMultiScale(
        eyesFrame,
        1.2,                        # scaleFactor
        3,                          # minNeighbors
        0|cv2.CASCADE_SCALE_IMAGE,  # flag
        (0,0))                      # minSize
    
    # If two eyes are detected, check norm of difference between pixels of eyes
    if (len(eyes) == 2):
    
        # Get detected values from eyes
        x1, y1, w1, h1 = eyes[0]
        x2, y2, w2, h2 = eyes[1]
        
        # Get max width and height between two eyes
        newW = max([w1, w2])
        newH = max([h1, h2])
        
        # Get eyes
        eye1Range = eyesFrame[y1:y1+h1, x1:x1+w1]
        eye2Range = eyesFrame[y2:y2+h2, x2:x2+w2]
        
        # Resize eyes to be the same size
        eye1Resize = cv2.resize(eye1Range, (newH, newW), interpolation = cv2.INTER_AREA)
        eye2Resize = cv2.resize(eye2Range, (newH, newW), interpolation = cv2.INTER_AREA)
        
        # Calculate norm of difference
        norm_diff = np.linalg.norm(eye1Resize - eye2Resize) / (newW * newH)
    
        # If norm between eye pixels is less than 2, then the face is not winking
        if norm_diff <= 2:
            return False
        
    # If only one eye is detected, then the face is winking
    elif len(eyes) == 1:
        return True
    
    #""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Fifth stage -> Use detect multi scale
    
    # Cascade classifier
    eyes = eyesCascade.detectMultiScale(
        eyesFrame,
        1.16,                       # scaleFactor
        2,                          # minNeighbors
        0|cv2.CASCADE_SCALE_IMAGE,  # flag
        (0,0))                      # minSize
    
    # If only one eye is detected, then the face is winking
    if len(eyes) == 1:
        return True
    
    #""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Sixth stage -> Use detect multi scale
    
    # Use histogram equalization on the face
    eyesFrame = cv2.equalizeHist(eyesFrame)
    
    # Cascade classifier
    eyes = eyesCascade.detectMultiScale(
        eyesFrame,
        1.01,                       # scaleFactor
        3,                          # minNeighbors
        0|cv2.CASCADE_SCALE_IMAGE,  # flag
        (0,0))                      # minSize
    
    # If one eye is detected, the face is winking, otherwise the face is not winking
    return len(eyes) == 1