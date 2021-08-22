'''
LuvHistogramEqualizationOpenCVWindow.py
	
	Description:
		This program takes a color image and a window of the image as inputs, performs histogram
			equalization in the Luv domain, and writes the scaled image as an output using OpenCV
			functions. It also displays the input image and the scaled output image.
'''

import cv2
import numpy as np
import sys
import math

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
inputName = sys.argv[5]
outputName = sys.argv[6]

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(inputName, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", inputName)
    sys.exit()

cv2.imshow("input image: " + inputName, inputImage)

rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

# Initialize window
window = np.zeros([int(H2)+1 - int(H1), int(W2)+1 - int(W1), 3], dtype='uint8')

# Get window pixel values
for i in range(int(H1), int(H2)+1) :
    for j in range(int(W1), int(W2)+1) :
        window[i - int(H1), j - int(W1)] = inputImage[i, j]

# Convert window from BGR image to Luv image
windowInLuvImage = cv2.cvtColor(window, cv2.COLOR_BGR2Luv)

# Get L values from window in Luv domain
LValues = np.zeros([int(H2)+1 - int(H1), int(W2)+1 - int(W1), 1], dtype='uint8')
for i in range(0, int(H2)+1 - int(H1)) :
    for j in range(0, int(W2)+1 - int(W1)) :
        L,u,v = windowInLuvImage[i, j]
        LValues[i, j] = L

# Perform histogram equalization on the window's L values
equalizedLuvValues = cv2.equalizeHist(LValues)

# Loop through window pixels and set new L values
for i in range(0, int(H2)+1 - int(H1)) :
    for j in range(0, int(W2)+1 - int(W1)) :
        L,u,v = windowInLuvImage[i, j]
        windowInLuvImage[i, j] = [equalizedLuvValues[i, j], u, v]

# Convert the window back to BGR
equalizedWindow = cv2.cvtColor(windowInLuvImage, cv2.COLOR_Luv2BGR)

# Construct the output image
outputImage = np.copy(inputImage)
for i in range(int(H1), int(H2)+1) :
    for j in range(int(W1), int(W2)+1) :
        b, g, r = equalizedWindow[i - int(H1), j - int(W1)]
        outputImage[i, j] = [b,g,r]

cv2.imshow("p4:", outputImage)
cv2.imwrite(outputName, outputImage);


# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
