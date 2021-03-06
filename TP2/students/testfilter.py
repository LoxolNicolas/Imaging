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

    sigma = 50 
    noise = np.zeros_like(img)
    cv2.randn(noise,(0),(sigma))
    img = img + noise

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
    WG5D = cv2.getGaussianKernel(20, 20)
    WG7D = cv2.getGaussianKernel(kernel_size, 20)

    WG2D = np.dot(WG1D, np.transpose(WG1D)) #PRODUIT MATRICIEL
    WG4D = np.dot(WG3D, np.transpose(WG3D)) #PRODUIT MATRICIEL
    WG6D = np.dot(WG5D, np.transpose(WG5D)) #PRODUIT MATRICIEL
    WG8D = np.dot(WG7D, np.transpose(WG7D)) #PRODUIT MATRICIEL

    # Apply the filter
    imfm = cv2.filter2D(img, cv2.CV_32F, W)
    imfm5 = cv2.filter2D(img, cv2.CV_32F, W5) #APPARITION DE FLOU

    imfg = cv2.filter2D(img, cv2.CV_32F, WG2D)
    imfg1 = cv2.filter2D(img, cv2.CV_32F, WG4D)
    imfg2 = cv2.filter2D(img, cv2.CV_32F, WG6D)
    imfg3 = cv2.filter2D(img, cv2.CV_32F, WG8D)

    imfcx = cv2.filter2D(img, cv2.CV_32F, Wx)
    imfcy = cv2.filter2D(img, cv2.CV_32F, Wy)

    imfcx_filtre = cv2.filter2D(imfg, cv2.CV_32F, Wx)
    imfcy_filtre = cv2.filter2D(imfg, cv2.CV_32F, Wy)

    imfcx_y = abs(imfcx) + abs(imfcy)

    #norme_img = norme(imfcx, imfcy)
    #theta_img = theta(imfcx, imfcy)

    norme_img = np.sqrt(imfcx**2 + imfcy**2)

    theta_img = np.arctan2(imfcy, imfcx)

    norme_img_filtre_gaussien = np.sqrt(imfcx_filtre**2 + imfcy_filtre**2)

    # Apply median filter
    imfm1 = cv2.medianBlur(img, 3)
    imfm2 = cv2.medianBlur(img, 5)
    imfm3 = cv2.medianBlur(img, 7)

    #Laplacian filter

    Wl1 = np.array([\
        (0, -1, 0),\
        (-1, 4, -1),\
        (0, -1, 0)]\
        ,dtype = float)


    Wl2 = np.array([\
        (-1, -1, -1),\
        (-1, 8, -1),\
        (-1, -1, -1)]\
        ,dtype = float)

    imfl1 = cv2.filter2D(img, cv2.CV_32F, Wl1)
    imfl2 = cv2.filter2D(img, cv2.CV_32F, Wl2)
    imfl3 = cv2.Laplacian(img, cv2.CV_32F, ksize = 3)

    ## Display  
    # image

    filenameexo = "im1.bmp"
    imexo = cv2.imread(filenameexo)
    imgexo = cv2.cvtColor(imexo, cv2.COLOR_BGR2GRAY)

    imgexo_x = cv2.filter2D(imgexo, cv2.CV_32F, Wx)
    imgexo_y = cv2.filter2D(imgexo, cv2.CV_32F, Wy)

    imgexo_xy = abs(imgexo_x) + abs(imgexo_y)

    plt.subplot(221) 
    plt.imshow(imgexo, 'gray',vmin=0, vmax=255) 

    for i in range(0, 20):
        imgexo = cv2.filter2D(imgexo, cv2.CV_32F, WG2D)
        
    th = 170

    imgb = 255 * (imgexo>th) + 0 * (imgexo*th)

    #imfxexo = cv2.filter2D(imgb, cv2.CV_32F, Wx)
    #imfyexo = cv2.filter2D(imgb, cv2.CV_32F, Wy)

    #normeexo = np.sqrt(imfxexo**2 + imfyexo**2)

    plt.subplot(222) 
    plt.imshow(imgexo_xy, 'gray', vmin=0, vmax=255)

    plt.subplot(223) 
    plt.imshow(imgb, 'gray', vmin=0, vmax=255)

    #plt.subplot(224) 
    #plt.imshow(normeexo, 'gray', vmin=0, vmax=255)

    #Appliquer plusieur fois un filtre gaussien sur l'image
    #Appliquer un masque th = 170
    # imgb = 255 * (img>th) +  0*(img*th)
    # Puis calcul du gradient
    # astype uint8 si pb

    plt.show()
 
if __name__ == "__main__":
    # execute only if run as a script
    main()
