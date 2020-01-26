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

def masque(img, ind_max):
	#FORMULE Image_sortie = Image * (1 - Image_masque) + Ih * Image_masque
	#Blanc pour tapis noir sinon

	masque = np.copie(img)

	for x in range(masque.shape[0]):
		for y in range(masque.shape[1]):
			if(masque[x][y][0] < ind_max + 20 and masque[x][y][0] > ind_max - 20):
				imghsv[x][y][0] = 1
			else:
				imghsv[x][y][0] = 0
	
	return masque


def main():
	img = cv2.imread('billard_large.jpg')
	imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	imghsv = cv2.cvtColor(imgrgb, cv2.COLOR_RGB2HSV)

	# Extract H, V, V channels
	# ... to be completed

	Ih = imghsv[:, :, 0]
	Is = imghsv[:, :, 1]
	Iv = imghsv[:, :, 2]

	# Compute hue histogram
	# ... to be completed
	# Find hue value for maximum of histogram
	# ... to be completed

	histo_hue = histo(Ih, 256)

	ind_max = np.where(histo_hue == max(histo_hue))[0][0]

	print(ind_max)

	for x in range(imghsv.shape[0]):
		for y in range(imghsv.shape[1]):
			if(imghsv[x][y][0] < ind_max + 20 and imghsv[x][y][0] > ind_max - 20):
				imghsv[x][y][0] = 0.4 * 255

	nimg = cv2.cvtColor(imghsv, cv2.COLOR_HSV2RGB)
			

	# Build a mask image
	# ... to be completed
	# Change green hue with blue one (about int(0.4*255))
	# ... to be completed
	#hnew = ...
	#imghsv[:,:,0]=hnew
	# Convert to RGB image
	# ... to be completed
	#imgrgbnew = ...

	# Display
	plt.subplot(321)

	plt.imshow(imgrgb)

	plt.title('Colour image')

	plt.subplot(322)

	plt.imshow(Ih, 'gray')

	plt.title('Hue')

	plt.subplot(323)

	plt.imshow(Is, 'gray')

	plt.title('Saturation')

	plt.subplot(324)

	plt.imshow(Iv, 'gray')

	plt.title('Brightness')

	plt.subplot(325)

	plt.plot(histo_hue)

	plt.title('Histo hue')

	plt.subplot(326)

	plt.imshow(nimg)

	plt.title('img_to_blue')

	# ... to be completed
	# ...
	plt.show()

if __name__ == "__main__":
    # execute only if run as a script
    main()
