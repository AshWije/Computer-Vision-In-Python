'''
LuvHistogramEqualization.py
	
	Description:
		This program takes a color image and a window of the image as inputs, performs histogram
			equalization in the Luv domain, and writes the scaled image as an output. It also
			displays the input image and the scaled output image.
'''

import cv2
import numpy as np
import sys
import math

def convertBGR2NonLinearRGB(b,g,r):
	# Calculate NonLinearRGB by dividing by 255
	Rnl = r / 255.0
	Gnl = g / 255.0
	Bnl = b / 255.0
	
	# Check bounds
	if Rnl < 0: Rnl = 0.0
	elif Rnl > 1: Rnl = 1.0
	
	if Gnl < 0: Gnl = 0.0
	elif Gnl > 1: Gnl = 1.0
	
	if Bnl < 0: Bnl = 0.0
	elif Bnl > 1: Bnl = 1.0
	
	return Rnl, Gnl, Bnl

def convertNonLinearRGB2LinearRGB(Rnl,Gnl,Bnl):
	# Calculate Rl using Inverse Gamma Correction
	if Rnl < 0.03928: Rl = Rnl / 12.92
	else: Rl = ((Rnl + 0.055) / 1.055) ** 2.4
	
	# Calculate Gl using Inverse Gamma Correction
	if Gnl < 0.03928: Gl = Gnl / 12.92
	else: Gl = ((Gnl + 0.055) / 1.055) ** 2.4
	
	# Calculate Bl using Inverse Gamma Correction
	if Bnl < 0.03928: Bl = Bnl / 12.92
	else: Bl = ((Bnl + 0.055) / 1.055) ** 2.4
	
	# Check bounds
	if Rl < 0: Rl = 0.0
	elif Rl > 1: Rl = 1.0
	
	if Gl < 0: Gl = 0.0
	elif Gl > 1: Gl = 1.0
	
	if Bl < 0: Bl = 0.0
	elif Bl > 1: Bl = 1.0
	
	return Rl,Gl,Bl

def convertLinearRGB2XYZ(Rl,Gl,Bl):
	# Calculate XYZ using Linear Transform
	X = (0.412453 * Rl) + (0.35758 * Gl) + (0.180423 * Bl)
	Y = (0.212671 * Rl) + (0.71516 * Gl) + (0.072169 * Bl)
	Z = (0.019334 * Rl) + (0.119193 * Gl) + (0.950227 * Bl)
	
	# Check bounds
	if X < 0: X = 0.0
	if Y < 0: Y = 0.0
	if Z < 0: Z = 0.0
	
	return X,Y,Z

def convertXYZ2Luv(X,Y,Z):
	# White values
	(Xw, Yw, Zw) = (0.95, 1.0, 1.09)
	
	# Calculate uw and vw
	uw = (4.0 * Xw) / (Xw + (15.0 * Yw) + (3.0 * Zw))
	vw = (9.0 * Yw) / (Xw + (15.0 * Yw) + (3.0 * Zw))
	
	# Calculate L
	t = Y / Yw
	if t > 0.008856: L = (116.0 * (t ** (1.0 / 3.0))) - 16.0
	else: L = 903.3 * t
	
	# Check bounds of L
	if L < 0: L = 0.0
	elif L > 100: L = 100.0
	
	# Calculate u and v
	d = X + (15.0 * Y) + (3.0 * Z)
	ux = 0.0
	vx = 0.0
	if d != 0:
		ux = (4.0 * X) / d
		vx = (9.0 * Y) / d
	u = 13.0 * L * (ux - uw)
	v = 13.0 * L * (vx - vw)
	
	return L,u,v
	
def convertLuv2XYZ(L,u,v):
	# White values
	(Xw, Yw, Zw) = (0.95, 1.0, 1.09)
	
	# Calculate uw and vw
	uw = (4.0 * Xw) / (Xw + (15.0 * Yw) + (3.0 * Zw))
	vw = (9.0 * Yw) / (Xw + (15.0 * Yw) + (3.0 * Zw))
	
	# Calculate ux and vx
	ux = 0.0
	vx = 0.0
	if L != 0:
		ux = (u + (13.0 * uw * L)) / (13.0 * L)
		vx = (v + (13.0 * vw * L)) / (13.0 * L)
	
	# Calculate Y
	if L > 7.9996: Y = (((L + 16.0) / 116.0) ** 3) * Yw
	else: Y = (L / 903.3) * Yw
	
	# Calculate X and Z
	X = 0.0
	Z = 0.0
	if vx != 0:
		X = (Y * 2.25) * (ux / vx)
		Z = (Y * (3.0 - (0.75 * ux) - (5.0 * vx))) / vx
	
	# Check bounds
	if X < 0: X = 0.0
	if Y < 0: Y = 0.0
	if Z < 0: Z = 0.0
	
	return X,Y,Z

def convertXYZ2LinearRGB(X,Y,Z):
	# Calculate LinearRGB using Linear Transform
	Rl = (3.240479 * X) + (-1.53715 * Y) + (-0.498535 * Z)
	Gl = (-0.969256 * X) + (1.875991 * Y) + (0.041556 * Z)
	Bl = (0.055648 * X) + (-0.204043 * Y) + (1.057311 * Z)
	
	# Check bounds
	if Rl < 0: Rl = 0.0
	elif Rl > 1: Rl = 1.0
	
	if Gl < 0: Gl = 0.0
	elif Gl > 1: Gl = 1.0
	
	if Bl < 0: Bl = 0.0
	elif Bl > 1: Bl = 1.0
	
	return Rl,Gl,Bl

def convertLinearRGB2NonLinearRGB(Rl,Gl,Bl):
	# Calculate Rl using Gamma Correction
	if Rl < 0.00304: Rnl = 12.92 * Rl
	else: Rnl = (1.055 * (Rl ** (1.0 / 2.4))) - 0.055
	
	# Calculate Gl using Gamma Correction
	if Gl < 0.00304: Gnl = 12.92 * Gl
	else: Gnl = (1.055 * (Gl ** (1.0 / 2.4))) - 0.055
	
	# Calculate Bl using Gamma Correction
	if Bl < 0.00304: Bnl = 12.92 * Bl
	else: Bnl = (1.055 * (Bl ** (1.0 / 2.4))) - 0.055
	
	# Check bounds
	if Rnl < 0: Rnl = 0.0
	elif Rnl > 1: Rnl = 1.0
	
	if Gnl < 0: Gnl = 0.0
	elif Gnl > 1: Gnl = 1.0
	
	if Bnl < 0: Bnl = 0.0
	elif Bnl > 1: Bnl = 1.0
	
	return Rnl,Gnl,Bnl

def convertNonLinearRGB2BGR(Rnl,Gnl,Bnl):
	# Calculate BGR by multiplying by 255
	r = 255.0 * Rnl
	g = 255.0 * Gnl
	b = 255.0 * Bnl
	
	# Check bounds
	if r > 255: r = 255.0
	elif r < 0: r = 0.0
	
	if g > 255: g = 255.0
	elif g < 0: g = 0.0
	
	if b > 255: b = 255.0
	elif b < 0: b = 0.0
	
	return round(b),round(g),round(r)

def convertLuv2BGR(L,u,v):
	# Luv -> XYZ -> LinearRGB -> NonLinearRGB -> BGR
    X,Y,Z = convertLuv2XYZ(L,u,v)
    Rl,Gl,Bl = convertXYZ2LinearRGB(X,Y,Z)
    Rnl,Gnl,Bnl = convertLinearRGB2NonLinearRGB(Rl,Gl,Bl)
    b,g,r = convertNonLinearRGB2BGR(Rnl,Gnl,Bnl)
    return b,g,r

def convertBGR2Luv(b,g,r):
	# BGR -> NonLinearRGB -> LinearRGB -> XYZ -> Luv
	Rnl,Gnl,Bnl = convertBGR2NonLinearRGB(b,g,r)
	Rl,Gl,Bl = convertNonLinearRGB2LinearRGB(Rnl,Gnl,Bnl)
	X,Y,Z = convertLinearRGB2XYZ(Rl,Gl,Bl)
	L,u,v = convertXYZ2Luv(X,Y,Z)
	return L,u,v

def histogramEqualizationInLuvDomain(h, totalNumPixels):
	# Initialize the new H array
	newH = np.zeros(101, dtype='uint64')
	
	# Calculate the new H values
	fPrev = 0
	for i in range(0, 101):
		f = fPrev + h[i]
		newH[i] = math.floor(((fPrev + f) / 2.0) * (101.0 / totalNumPixels))
		
		# Check bounds
		if newH[i] < 0: newH[i] = 0
		if newH[i] > 100: newH[i] = 100
		fPrev = f
		
	return newH
	
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

# Initialize histogram
h = np.zeros(101, dtype='uint64')

# Calculate the total number of pixels
totalNumPixels = (int(H2)+1 - int(H1)) * (int(W2)+1 - int(W1))

# Set histogram values
for i in range(int(H1), int(H2)+1) :
    for j in range(int(W1), int(W2)+1) :
        b, g, r = inputImage[i, j]
        L, u, v = convertBGR2Luv(b,g,r)
        h[int(math.floor(L))] = h[int(math.floor(L))] + 1

# Get the new histogram after equalization
newH = histogramEqualizationInLuvDomain(h, totalNumPixels)

# Loop through the window and create the new equalized image
outputImage = np.copy(inputImage)
for i in range(int(H1), int(H2)+1) :
    for j in range(int(W1), int(W2)+1) :
        b, g, r = inputImage[i, j]
        L, u, v = convertBGR2Luv(b,g,r)
        b,g,r = convertLuv2BGR(newH[int(math.floor(L))], u, v)
        outputImage[i, j] = [b,g,r]

cv2.imshow("p3:", outputImage)
cv2.imwrite(outputName, outputImage);


# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
