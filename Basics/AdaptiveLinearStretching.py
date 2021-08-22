# AdaptiveLinearStretching.py
#
# Description:
#   This program gets as inputs a window parameter (w) and names of the input and the
#   output images. The output image is computed from the input image by changing only
#   the L values such that L(i, j) is computed by applying linear stretching to the
#   window of size (2w+1 × 2w+1) centered at (i, j). The L value at the center of the
#   window is the output value L(i, j). It should be executed as follows:
#       python AdaptiveLinearStretching.py <w> <inputImage> <outputImage>


import cv2
import numpy as np
import sys

# Read arguments
if(len(sys.argv) != 4) :
    print(sys.argv[0], ": takes 3 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w ImageIn ImageOut.")
    print("Example:", sys.argv[0], "10 fruits.jpg out.png")
    sys.exit()

w = int(sys.argv[1])
inputName = sys.argv[2]
outputName = sys.argv[3]

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

# Initialize output image
newLuvImage = np.zeros([rows, cols, bands],dtype=np.uint8)

# Note: Output pixel is computed from a window of size 2w+1 x 2w+1 in the input image
# Loop through windows
for i in range(w, rows-w) :
    for j in range(w, cols-w) :
        # Get array of all L values in window
        L_values = luvImage[i-w:i+w+1, j-w:j+w+1, 0].flatten()

        # Find the min and max L values
        a = min(L_values) # Min L value
        b = max(L_values) # Max L value

        # Get L, u, and v values at pixel (i,j)
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



