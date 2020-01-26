#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 08:46:29 2018
testfilter is the "to be completed" script for practical 2
part 2 of computer vision lectures 
@author: thierrychateau
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np

# This is the main function that is launched when this file is run as a script
def main():
    # Read an image file 
    filename = "scene.png"
    im = cv2.imread(filename)
    # Convert to grayscale image
    img=cv2.cvtColor( im, cv2.COLOR_BGR2GRAY)
    # convert to binary image
    threshold = 70 
    ret,imb = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY_INV)
# apply operation
    # define kernel
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((7,7),np.uint8)
    erosion1 = cv2.erode(imb,kernel,iterations = 3)
    erosion2 = cv2.erode(imb,kernel2,iterations = 1)
    opening = cv2.morphologyEx(imb, cv2.MORPH_OPEN, kernel)
    ## Display  
    # image
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.subplot(131) 
    plt.imshow(imb, 'gray')
    plt.title('Input image')   
    plt.subplot(132) 
    plt.imshow(erosion1, 'gray')
    plt.title(' Eroded image 1')
    plt.subplot(133) 
    plt.imshow(erosion2, 'gray')
    plt.title(' Eroded image 2')
    plt.show()
 
if __name__ == "__main__":
    # execute only if run as a script
    main()
