# -*- coding: utf-8 -*-
import numpy as np 
import cv2 as cv
import urllib
from .service_cal_distance import main
from .service_detect_curve import detect_curve
kernel = np.ones((3,1), np.uint8)
kernel2 = np.ones((1,3), np.uint8)

def service_remove_noise(PATH,actual_external_diameter):    
    image = cv.imread(PATH)
    img = cv.resize(image, dsize=(1280, 756),interpolation=cv.INTER_AREA)
    grab_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # blurred = cv.GaussianBlur(grab_gray, (5, 5), 0)
    thresh = cv.threshold(grab_gray, 100, 255, cv.THRESH_BINARY)[1]
    soble_bitwise = cv.bitwise_not(thresh)
    result = cv.erode(soble_bitwise, kernel, iterations = 10)
    contours, _ = cv.findContours(result.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img2 = img.copy()
    c = max(contours, key=cv.contourArea)
    x,y,w,h = cv.boundingRect(c)
    roi_image = img[y:y+h,x:x+w]

    roi_gray = cv.cvtColor(roi_image, cv.COLOR_BGR2GRAY)
    roi_mask = cv.threshold(roi_gray, 100, 255, cv.THRESH_BINARY)[1]
    mask_inv = cv.bitwise_not(thresh)

    img2 = cv.bitwise_and(img2, img2, mask=mask_inv)
    curve_img = img2.copy()
    # curve_result = detect_curve(curve_img)
    depth_result = main(img2, actual_external_diameter)
    curve = detect_curve(curve_img)
    return depth_result, curve
    



   

