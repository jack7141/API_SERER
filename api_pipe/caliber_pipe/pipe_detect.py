# -*- coding: utf-8 -*-
import numpy as np 
import cv2 as cv
import urllib
from .pipe_caliber import main
kernel = np.ones((3,1), np.uint8)
kernel2 = np.ones((1,3), np.uint8)

def opencv_pipe_detect(PATH):    
    image = cv.imread(PATH)
    img = cv.resize(image, dsize=(1280, 756),interpolation=cv.INTER_AREA)
    #STEP 2: Convert to Gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #STEP 3: Extract Canny 
    edged = cv.Canny(gray, 0, 255)
    #STEP 4: Filter Sobel
    img_sobel_y = cv.Sobel(edged, cv.CV_64F, 0, 1, ksize=13)
    img_sobel_y = cv.convertScaleAbs(img_sobel_y)
    # ADD
    img_sobel_y = cv.dilate(img_sobel_y, kernel2, iterations = 5)
    img_sobel_y = cv.erode(img_sobel_y, kernel2, iterations = 5)

    #STEP 4: Reverse binary image 
    soble_bitwise = cv.bitwise_not(img_sobel_y)
    result = cv.dilate(soble_bitwise, kernel, iterations = 11)
    contours, _ = cv.findContours(result.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img2 = img.copy()
    c = max(contours, key=cv.contourArea)
    x,y,w,h = cv.boundingRect(c)
    roi_image = img[y:y+h,x:x+w]
    rect = (x,y,x+w,y+h)
    mask = np.zeros(img2.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    try:
        cv.grabCut(img2,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    except cv.error as e:
        print("GrabCut Error")
        return None, None
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img2 = img2*mask2[:,:,np.newaxis]
    distance_result = main(img2)
    return distance_result
    



   

