#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 08:46:29 2018
testharris is the "to be completed" script for practical 3
part 2 of computer vision lectures 
@author: thierrychateau
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np
#import pdb ; pdb.set_trace()
# This is the main function that is launched when this file is run as a script
#def to_kpts(pts, size=1): 
#        return [cv2.KeyPoint(pt[0], pt[1], size) for pt in pts]
def main():
    # Read an image file 
    filenameref = "toys.jpg"
    filenametest = "toy_rot.jpg"
    imref = cv2.imread(filenameref)
    imtest = cv2.imread(filenametest)
    # Convert to grayscale image
    imgref=cv2.cvtColor( imref, cv2.COLOR_BGR2GRAY )
    imrefRGB=cv2.cvtColor( imref, cv2.COLOR_BGR2RGB )
    imgtest=cv2.cvtColor( imtest, cv2.COLOR_BGR2GRAY )
    imtestRGB=cv2.cvtColor( imtest, cv2.COLOR_BGR2RGB )
    # Compute Interest points using opencv function
    #pref = cv2.goodFeaturesToTrack(imgref,maxCorners = 300, qualityLevel = 0.01, minDistance = 4, useHarrisDetector=True, k=0.04)
    #pref2 = np.float32(pref).reshape(-1, 2)
    # Compute interest points and descriptors
    #alg = cv2.xfeatures2d.SIFT_create()
    alg = cv2.xfeatures2d.SURF_create()
    #pdb.set_trace();
    kpr, dr=alg.detectAndCompute(imgref,None)
    kpt, dt=alg.detectAndCompute(imgtest,None)
    imrefRGB = cv2.drawKeypoints(imrefRGB, kpr, imrefRGB, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imtestRGB = cv2.drawKeypoints(imtestRGB, kpt, imrefRGB, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors.
    matches = bf.match(dr,dt)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(imrefRGB,kpr,imtestRGB,kpt,matches[:10], None, flags=2)
    ## Display  
    # image
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    #plt.subplot(321) 
    plt.imshow(img3)
    plt.show()
 
if __name__ == "__main__":
    # execute only if run as a script
    main()

