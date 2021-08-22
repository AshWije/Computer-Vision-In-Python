'''
CreateNegativeImage.py
	
	Description:
		This program takes an image and creates the negative image for it. It also displays the
			original image, gray-level image, and negative image.
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
    print sys.argv[0], "failed to read image from", inputName
    sys.exit()
	
# Display the original image
cv2.imshow('original image', inputImage);

# If the image is color, convert it to a gray-level image
rank = len(inputImage.shape)
if(rank == 2) :
    gray_image = inputImage
elif(rank == 3) :
    gray_image = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
else :
    print sys.argv[0], "can't handle unusual image", inputName
    sys.exit()

# Display the gray-level image
cv2.imshow('gray image', gray_image);

# Initialize image output
rows, cols = gray_image.shape
image_output = np.zeros([rows, cols], dtype=np.uint8)

# Compute the negative of the gray-level image
for i in range(0, rows) :
    for j in range(0, cols) :
        image_output[i,j] = 255 - gray_image[i,j]

# Display the negative image
cv2.imshow('negative image', image_output);

# Write the negative image
cv2.imwrite(outputName, image_output);

# Wait for any key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()

