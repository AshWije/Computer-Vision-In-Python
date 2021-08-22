'''
CreateGradientImage.py
	
	Description:
		This program takes a width and height as inputs and creates an image with continuous changes
			in color from the Luv representation. It also displays this created image.
'''

import cv2
import numpy as np
import sys

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

    
if(len(sys.argv) != 3) :
    print(sys.argv[0], ": takes 2 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: width height.")
    print("Example:", sys.argv[0], " 200 300")
    sys.exit()

cols = int(sys.argv[1])
rows = int(sys.argv[2])

image = np.zeros([rows, cols, 3], dtype='uint8') # Initialize the image with all 0
for i in range(0, rows):
    for j in range (0,cols):
		# 0<=L<=100, -134<=u<=220, -140<=v<=122 
        L = 90      
        u = (354 * j / cols) - 134
        v = (262 * i / rows) - 140
        b,g,r = convertLuv2BGR(L,u,v)
        image[i,j]=np.array([b,g,r],dtype='uint8')
    
cv2.imshow("p1:", image)


# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
