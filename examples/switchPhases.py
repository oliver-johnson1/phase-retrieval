import numpy as np
from imageio import imread
from skimage import color
import matplotlib.pyplot as plt
import skimage.io as io

image1 = imread('data/BSDS300/41004.jpg') 
image1 = color.rgb2gray(image1)

image2 = imread('data/BSDS300/67079.jpg') 
image2 = color.rgb2gray(image2)

mag1 = np.abs(np.fft.fft2(image1))
phase1 = np.angle(np.fft.fft2(image1))

mag2 = np.abs(np.fft.fft2(image2))
phase2 = np.angle(np.fft.fft2(image2))

# reconstruct image but switching phases
image_fft1 = mag1*np.exp(1j*phase2)
image_fft2 = mag2*np.exp(1j*phase1)

image_reconstructed1 = np.real(np.fft.ifft2(image_fft1))
image_reconstructed2 = np.real(np.fft.ifft2(image_fft2))

io.imsave('outputs/image1.png', np.clip(image1*255, a_min=0, a_max=255.))
io.imsave('outputs/image2.png', np.clip(image2*255, a_min=0, a_max=255.))
io.imsave('outputs/image_reconstructed1.png', np.clip(image_reconstructed1*255, a_min=0, a_max=255.))
io.imsave('outputs/image_reconstructed2.png', np.clip(image_reconstructed2*255, a_min=0, a_max=255.))