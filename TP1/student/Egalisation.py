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

def histo_cumule(h):
    h_cumul = np.zeros(h.shape[0])
    h_cumul[0] = h[0]
    for i in range(1, h.shape[0]):
        h_cumul[i] = h[i] + h_cumul[i - 1]
    return h_cumul

def nb_pixel_histo(h):
    nb = 0
    for i in range(1, h.shape[0]):
        nb = nb + h[i]
    return nb

def histo_cumule_egali(h_cumul, taille):
    h_cumul_egali = np.zeros(h_cumul.shape[0])
    h_cumul_egali[0] = h_cumul[0] / taille
    for i in range(1, h_cumul.shape[0]):
        h_cumul_egali[i] = int((h_cumul[i] / taille) * 255)
        print(h_cumul_egali[i])
    return h_cumul_egali

def histo_egalisation(h_cumul_egali, h):
    h_egal = np.zeros(h_cumul_egali.shape[0])
    for i in range(0, h_cumul_egali.shape[0]):
        h_egal[int(h_cumul_egali[i])] = h_cumul[i]
    return h_egal

def main():
    filename = "mountns1.jpg"
    img = cv2.imread(filename)
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h = histo(imggray, 256)
    h_cumul = histo_cumule(h)

    h_cumul_egali = histo_cumule_egali(h_cumul, nb_pixel_histo(h))

    h_egal = histo_egalisation(h_cumul_egali, h)
    
    plt.subplot(221) 
    plt.imshow(imggray, 'gray',vmin=0, vmax=255)
    plt.title('Grayscale image')

    plt.subplot(222) 
    plt.plot(h)
    plt.title('Histogram image')

    plt.subplot(223) 
    plt.plot(h_egal)
    plt.title('Histogram image')

    plt.show()
#    
if __name__ == "__main__":
    # execute only if run as a script
    main()
