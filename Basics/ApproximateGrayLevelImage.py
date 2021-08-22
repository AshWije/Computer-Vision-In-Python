'''
ApproximateGrayLevelImage.py
	
	Description:
		This program takes an image and creates an approximate gray-level image for it using the
			best of four methods (y=g, y=max{r,g,b}, y=round((r+g+b)/3.0), y=round(0.3r+0.6g+0.1b)).
			It also displays the original image, the real gray-level image, and the four approximate
			gray-level images (using the four methods).
			
	Analysis:
		Method 1: y = g
			Approximate gray-level image depends on the amount of green present in the image.
				
		Method 2: y = max{r, g, b}
			Approximate gray-level image becomes brighter than it should be.
			
		Method 3: y = round((r + g + b) / 3.0)
			Approximate gray-level image darkens almost to the point of becoming the negative.
			
		Method 4: y = round(0.3r + 0.6g + 0.1b))
			Most accurate approximate gray-level image to the original image.
'''

import cv2
import numpy as np
import sys

# Make sure that two arguments are provided
if(len(sys.argv) != 3) :
    print "Incorrect number of arguments. Execute with: python", sys.argv[0], "'input-image-name' 'output-image-name'"
    sys.exit()

# Assign the arguments to variables
inputName = sys.argv[1]
outputName = sys.argv[2]

# Read the image
inputImage = cv2.imread(inputName, cv2.IMREAD_UNCHANGED);

# Make sure that the image exists
if(inputImage is None) :
    print sys.argv[0], "failed to read image from:", inputName
    sys.exit()

# Make sure the image is a standard color image
if(len(inputImage.shape) != 3 or inputImage.shape[2] != 3) :
    print sys.argv[0], "not a standard color image:", inputName 
    sys.exit()

# Display the original image
cv2.imshow('original image', inputImage);

# Initialize image outputs
rows, cols, bands = inputImage.shape # bands == 3
imageOutput1 = np.zeros([rows, cols], dtype=np.uint8)
imageOutput2 = np.zeros([rows, cols], dtype=np.uint8)
imageOutput3 = np.zeros([rows, cols], dtype=np.uint8)
imageOutput4 = np.zeros([rows, cols], dtype=np.uint8)

# Compute approximate gray-level images
for i in range(0, rows) :
    for j in range(0, cols) :
        b, g, r = inputImage[i, j]
        imageOutput1[i,j] = g
        imageOutput2[i,j] = max(r, g, b)
        imageOutput3[i,j] = round((r + g + b) / 3.0)
        imageOutput4[i,j] = round(0.3*r + 0.6*g + 0.1*b)

# Display actual gray-level image (for comparison)
grayImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray-level image', grayImage);

# Display the approximate gray-level images
cv2.imshow('output image 1: y = g', imageOutput1);
cv2.imshow('output image 2: y = max{r,g,b}', imageOutput2);
cv2.imshow('output image 3: y = round((r + g + b) / 3.0)', imageOutput3);
cv2.imshow('output image 4: y = round(0.3r + 0.6g + 0.1b))', imageOutput4);

# Write the best approximate gray-level image (method 4, see analysis above)
cv2.imwrite(outputName, imageOutput4);

# Wait for any key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()

