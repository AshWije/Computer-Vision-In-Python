# LuvHistogramEqualizationOpenCV.py
#
# Description:
#   This program takes the names of the input and the output images as inputs and then
#   computes the output image by histogram equalizing the input image applied to the L
#   values in the Luv color space. It should be executed as follows:
#       python LuvHistogramEqualizationOpenCV.py <inputImage> <outputImage>


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

# Perform Histogram Equalization on L
origL = luvImage[:, :, 0]
newL = cv2.equalizeHist(origL)

# Update Luv image with new L values
luvImage[:, :, 0] = newL

# Convert Luv image back to BGR image
outputImage = cv2.cvtColor(luvImage, cv2.COLOR_Luv2BGR)
cv2.imshow("output image", outputImage)

# saving the output 
cv2.imwrite(outputName, outputImage)

# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()