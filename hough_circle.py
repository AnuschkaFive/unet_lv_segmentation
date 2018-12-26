from pathlib import Path
import cv2
import numpy as np

img = cv2.imread('data/heart_scans/ALFE-BL/ALFE-BL_CineMR_ti10_sl04_ORIG.png', 0)
minPix = img.min()
maxPix = img.max()
a = 255 / (maxPix - minPix)
b = -1 * a * minPix
img = ((img*a)+b).astype('uint8')

#print(img.shape)

#img = cv2.imread('data/opencv_logo.png', 0)

#print(img.shape)
#cv2.imshow('original', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

img = cv2.GaussianBlur(img,(5,5),0)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100,
                           param1=100, param2=50, minRadius=0, maxRadius=int(360/6))

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    cv2.circle(img, (i[0], i[1]),i[2],(255,255,255),2)
    cv2.circle(img, (i[0], i[1]),2,(255,255,255),3)
    
cv2.imshow('circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()