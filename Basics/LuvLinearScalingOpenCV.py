# LuvLinearScalingOpenCV.py
#
# Description:
#   This program takes the names of the input and the output images as inputs and then
#   computes the output image by linearly stretching the L values in the Luv color space
#   from the input image to their fullest range. It should be executed as follows:
#       python LuvLinearScalingOpenCV.py <inputImage> <outputImage>


import cv2
import numpy as np
import sys

# Read arguments
if(len(sys.argv) != 3) :
    print(sys.argv[0], ": takes 2 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: ImageIn ImageOut.")
    print("Example:", sys.argv[0], "fruits.jpg out.png")
    sys.exit()

inputName = sys.argv[1]
outputName = sys.argv[2]

# Read image
inputImage = cv2.imread(inputName, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", inputName)
    sys.exit()
cv2.imshow("input image: " + inputName, inputImage)

rows, cols, bands = inputImage.shape
if(bands != 3) :
    print("Input image is not a standard color image:", inputImage)
    sys.exit()

# Convert BGR image to Luv image
luvImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2Luv)

# Get array of all L values in image
L_values = luvImage[:, :, 0].flatten()

# Make sure there is at least one L value
if (L_values.size == 0):
    print("Input image has zero pixels.")
    sys.exit()

# Find the min and max L values
a = min(L_values) # Min L value
b = max(L_values) # Max L value

# Initialize output image
newLuvImage = np.zeros([rows, cols, bands],dtype=np.uint8)

# Loop through pixels
for i in range(rows) :
    for j in range(cols) :
        # Get L, u, and v values from pixel
        origL, u, v = luvImage[i, j]
        
        # Perform linear stretching of L in the range [0, 255]
        if (a != b): newL = (origL - a) * 255 // (b - a)
        else: newL = origL
        newLuvImage[i, j] = [newL, u, v]

# Convert Luv image back to BGR image
outputImage = cv2.cvtColor(newLuvImage, cv2.COLOR_Luv2BGR)
cv2.imshow("output image", outputImage)

# Saving the output 
cv2.imwrite(outputName, outputImage)

# Wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()