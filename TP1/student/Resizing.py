#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from matplotlib import pyplot as plt
import numpy as np

def main():
    filename = "lena1.jpg"
    img = cv2.imread(filename)
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hauteur = img.shape[0]
    largeur = img.shape[1]

    scale_percent = 60 # percent of original size

    width_resize = int(img.shape[1] * scale_percent / 100)
    height_resize = int(img.shape[0] * scale_percent / 100) 

    dim_resize = (width_resize, height_resize) 

    img_resized = cv2.resize(img, dim_resize, interpolation = cv2.INTER_AREA) 
    
    plt.subplot(211) 
    plt.imshow(img, 'gray',vmin=0, vmax=255)
    plt.title('Grayscale image')

    plt.subplot(212) 
    plt.imshow(img_resized, 'gray',vmin=0, vmax=255)
    plt.title('Grayscale image')

    plt.show()
#    
if __name__ == "__main__":
    # execute only if run as a script
    main()
