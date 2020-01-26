#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from matplotlib import pyplot as plt
import numpy as np

def histo(im, nb):
    h = np.zeros(nb)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            #print(im[i, j])
            #h[im[i, j]] = h[im[i, j]] + 1 #Avec 256 Bins
            valeur = int((im[i, j] * nb) / 256)
            h[valeur] += 1
    return h

def etirement(img):
    mini = 256
    maxi = 0
    for i in img:
        for j in i:
            mini = min(j, mini)
            maxi = max(j, maxi)

    a = 255 / (maxi-mini)
    b = -a * mini
    im2 = a * img + b

    return im2
    
def etirement2(img):
    mini = 256
    maxi = 0
    for i in img:
        for j in i:
            mini = min(j, mini)
            maxi = max(j, maxi)

    img2 = np.copy(img)

    for x in range(img2.shape[0]):
        for y in range(img2.shape[1]):
            img2[x, y] = int(float(img2[x, y] - mini) / float(maxi - mini) * 255)
            
    return img2

def main():
    filename = "mountns1.jpg"
    img = cv2.imread(filename)
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h = histo(imggray, 256)
    h1 = cv2.calcHist([imggray], [0], None, [256], [0, 256])
    
    plt.subplot(221) 
    plt.imshow(imggray, 'gray',vmin=0, vmax=255)
    plt.title('Grayscale image')

    plt.subplot(222) 
    plt.plot(h)
    plt.title('Histogram image')

    plt.subplot(223) 
    plt.plot(h1)
    plt.title('Histogram image openCV')

    img_etir = etirement(imggray)

    histo_etir = histo(img_etir, 256)

    img_etir_2 = etirement2(imggray)

    imeq = cv2.equalizeHist(imggray)

    histo_etir = histo(img_etir_2, 256)

    plt.subplot(224)
    plt.imshow(img_etir_2, 'gray',vmin=0, vmax=255)
    plt.title('Image etire en utlisant opencv2')

    plt.show()
#    
if __name__ == "__main__":
    # execute only if run as a script
    main()
