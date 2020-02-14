#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from matplotlib import pyplot as plt
import numpy as np
# import pdb ; pdb.set_trace()

#Harris --> Detecteur de point
#1 --> Calcul du score de harris
#2 --> Seuillage (on conserve les scores important)
#3 --> NMS

def nms(S): #non maximal suppression (supprimer les maximals non locaux)

    W1 = np.array([\
        (1, 0, 0),\
        (0, -1, 0),\
        (0, 0, 0)]\
        ,dtype = float)

    C1 = cv2.filter2D(S, cv2.CV_32F, W1)<0 

    W2 = np.array([\
        (0, 1, 0),\
        (0, -1, 0),\
        (0, 0, 0)]\
        ,dtype = float)

    C2 = cv2.filter2D(S, cv2.CV_32F, W2)<0 

    W3 = np.array([\
        (0, 0, 1),\
        (0, -1, 0),\
        (0, 0, 0)]\
        ,dtype = float)

    C3 = cv2.filter2D(S, cv2.CV_32F, W3)<0 

    W4 = np.array([\
        (0, 0, 0),\
        (1, -1, 0),\
        (0, 0, 0)]\
        ,dtype = float)

    C4 = cv2.filter2D(S, cv2.CV_32F, W4)<0 

    W5 = np.array([\
        (0, 0, 0),\
        (0, -1, 1),\
        (0, 0, 0)]\
        ,dtype = float)

    C5 = cv2.filter2D(S, cv2.CV_32F, W5)<0 

    W6 = np.array([\
        (0, 0, 0),\
        (0, -1, 0),\
        (1, 0, 0)]\
        ,dtype = float)

    C6 = cv2.filter2D(S, cv2.CV_32F, W6)<0 

    W7 = np.array([\
        (0, 0, 0),\
        (0, -1, 0),\
        (0, 1, 0)]\
        ,dtype = float)

    C7 = cv2.filter2D(S, cv2.CV_32F, W7)<0 

    W8 = np.array([\
        (0, 0, 0),\
        (0, -1, 0),\
        (0, 0, 1)]\
        ,dtype = float)

    C8 = cv2.filter2D(S, cv2.CV_32F, W8)<0 

    Sf = C1 & C2 & C3 & C4 & C5 & C6 & C7 & C8

    return Sf

def main():

    filename = "toy1.jpg"

    im = cv2.imread(filename)

    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    Wgx = np.array([\
        (-1, 0, 1),\
        (-2, 0, 2),\
        (-1, 0, 1)]\
        ,dtype = float)

    Wf = np.ones((5,5))/25

    Ix = cv2.filter2D(img, cv2.CV_32F, Wgx)
    Iy = cv2.filter2D(img, cv2.CV_32F, np.transpose(Wgx))

    Ixx = Ix * Ix
    Iyy = Iy * Iy

    Ixy = Ix * Iy
    Iyx = Iy * Ix

    WIxx = cv2.filter2D(Ixx, cv2.CV_32F, Wf)
    WIyy = cv2.filter2D(Iyy, cv2.CV_32F, Wf)
    WIxy = cv2.filter2D(Ixy, cv2.CV_32F, Wf)
    WIyx = cv2.filter2D(Iyx, cv2.CV_32F, Wf)

    trace = WIxx + WIyy

    det = (WIxx * WIyy) - (WIxy - WIyx)

    K = 0.04
    S = det - (K * (trace * trace)) 

    # threshold according to a fraction of max scores
    #fr = 0.01
    #S2 = S * ( S > fr*np.max(S.flatten()))

    # or threshold according to a fix value
    th = 300000
    S2 = S * (S > th)

    #print("score max", np.max(S.flatten()))

    #non maxima suppression
    S3 = nms(S2)

    # get coordinates
    ptsl, ptsc  = np.nonzero(S3)

    #pdb.set_trace()

    ## Display  
    # image
    
    plt.subplot(321) 
    plt.imshow(img, 'gray',vmin=0, vmax=255)
    plt.title('original image')
    plt.plot(ptsc, ptsl, '+r')

    plt.subplot(322)
    plt.imshow(S) 

    # histo
    plt.subplot(323)
    plt.imshow(S2)
    plt.title('Harris score')
    
    #plt.subplot(323) 
    #plt.imshow(iGyy)
    #plt.title('iGyy')
    #plt.subplot(324) 
    #plt.imshow(S3)

    #plt.subplot(223) 
    #plt.imshow(imgst, 'gray',vmin=0, vmax=255)
    #plt.title('Stretched image')
    #plt.subplot(324) 
    # histo
    #plt.plot(hst)
    #plt.title('Histogram for stretched image')
    #plt.subplot(325) 
    #plt.imshow(imeq, 'gray',vmin=0, vmax=255)
    #plt.title('Equalized image')
    #plt.subplot(326) 
    # histo
    #plt.plot(heq)
    #plt.title('Histogram for stretched image')
    plt.show()
 
if __name__ == "__main__":
    # execute only if run as a script
    main()
