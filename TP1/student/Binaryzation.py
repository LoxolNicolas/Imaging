#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from matplotlib import pyplot as plt
import numpy as np

def binaryzation(img):
    img_binary = np.copy(img)
    for i in range(img_binary.shape[0]):
        for j in range(img_binary.shape[1]):
            if(img_binary[i][j] > 127):
                img_binary[i][j] = 255
            else:
                img_binary[i][j] = 0
    
    return img_binary

def main():
    filename = "lena1.jpg"
    img = cv2.imread(filename)
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    plt.subplot(211) 
    plt.imshow(img, 'gray',vmin=0, vmax=255)
    plt.title('Grayscale image')

    img_binary = binaryzation(imggray)

    plt.subplot(212) 
    plt.imshow(img_binary, 'gray',vmin=0, vmax=255)
    plt.title('Grayscale image')

    plt.show()
#    
if __name__ == "__main__":
    # execute only if run as a script
    main()
