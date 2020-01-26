#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

def theta(imgX, imgY):
    h = np.zeros(512*512)
    i = 0
    for x in range(imgX.shape[0]):
        for y in range(imgX.shape[1]):
            h[i] = math.atan2(imgY[x, y], imgX[x, y]) 
            i += 1
    return h

def norme(imgX, imgY):
    h = np.zeros(512*512)
    i = 0
    for x in range(imgX.shape[0]):
        for y in range(imgX.shape[1]):
            h[i] = math.sqrt(imgY[x, y]**2 + imgX[x, y]**2) 
            i += 1
    return h

def main():
    filename = "lena1.jpg"
    im = cv2.imread(filename)
    img=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Add noise to image
    # sigma of noise (additive gaussian noise)
    # You should modify sigma to change noise amplitude

    #sigma = 50 
    #noise = np.zeros_like(img)
    #cv2.randn(noise,(0),(sigma))
    #img = img + noise

    # Define some Kernels

    # Sobel 
    Wx = np.array([\
        (-1, 0, 1),\
        (-2, 0, 2),\
        (-1, 0, 1)]\
        ,dtype = float)

    Wy = np.array([\
        (-1, -2, -1),\
        (0, 0, 0),\
        (1, 2, 1)]\
        ,dtype = float)

    # Medium
    W = np.ones((5,5)) / 25
    W5 = np.ones((11, 11)) / 121

    # Gaussian
    kernel_size = 10
    sigma = 2
    WG1D = cv2.getGaussianKernel(kernel_size, sigma)
    WG3D = cv2.getGaussianKernel(200, sigma)
    WG4D = cv2.getGaussianKernel(20, 20)
    WG5D = cv2.getGaussianKernel(kernel_size, 20)
    #WG2D = WG1D @ np.transpose(WG1D)

    # Apply the filter
    imfm = cv2.filter2D(img, cv2.CV_32F, W)
    imfm5 = cv2.filter2D(img, cv2.CV_32F, W5) #APPARITION DE FLOU

    imfg = cv2.filter2D(img, cv2.CV_32F, WG1D)
    imfg1 = cv2.filter2D(img, cv2.CV_32F, WG3D)
    imfg2 = cv2.filter2D(img, cv2.CV_32F, WG4D)
    imfg3 = cv2.filter2D(img, cv2.CV_32F, WG5D)

    imfcx = cv2.filter2D(img, cv2.CV_32F, Wx)
    imfcy = cv2.filter2D(img, cv2.CV_32F, Wy)

    imfcx_y = abs(imfcx) + abs(imfcy)

    norme_img = norme(imfcx, imfcy)

    theta_img = theta(imfcx, imfcy)

    # Apply median filter
    imfm1 = cv2.medianBlur(img, 3)
    imfm2 = cv2.medianBlur(img, 5)
    imfm3 = cv2.medianBlur(img, 7)

    ## Display  
    # image
    
    #fig = plt.figure()
    #fig.patch.set_facecolor('white')
    plt.subplot(221) 
    plt.imshow(img, 'gray',vmin=0, vmax=255)
    plt.title('Input image')   

    plt.subplot(222) 
    plt.imshow(imfm1, 'gray', vmin=0, vmax=255)
    plt.title('Contour x')

    plt.subplot(223) 
    plt.imshow(imfm2 , 'gray', vmin=0, vmax=255)
    plt.title('Contour y')

    plt.subplot(224) 
    plt.imshow(imfm3 , 'gray', vmin=0, vmax=255)
    plt.title('COntour x et y')

    plt.show()
 
if __name__ == "__main__":
    # execute only if run as a script
    main()
