#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 08:46:29 2018
hello_wold simply load an image with opencv, convert it to RGB and GRAYSCALE 
and display it using matplotlib
@author: thierrychateau
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np
# This is an example of function with a string argument that return another
# string 
def my_func(my_arg):
    string2 = my_arg + ' toto ';
    return string2
# This is the main function that is launched when this file is run as a script
def main():
    # call function my_func
    st = my_func("my name is")
    print(st)
    # Read an image file (im is a numpy matrix 
    # (nblines,nbcolums, BGR 3 channels))
    filename = "billard_large.jpg"
    img = cv2.imread(filename)
    # Convert to RGB
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to grayscale image
    imggray=cv2.cvtColor( imgrgb, cv2.COLOR_RGB2GRAY )
    ## Display the RGB and gray images 
    # divide plot screen into 1 line, 2 columns and select first subscreen as 
    # current screen
    plt.subplot(121) 
    plt.imshow(imgrgb)
    plt.title('Color RGB image')
    plt.subplot(122) 
    plt.imshow(imggray, 'gray',vmin=0, vmax=255)
    plt.title('Grayscale image')
    plt.show()
#    
if __name__ == "__main__":
    # execute only if run as a script
    main()
