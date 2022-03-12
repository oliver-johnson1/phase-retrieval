import numpy as np
from imageio import imread
from skimage import color
import skimage.io as io

image1 = imread('data/birds.png') 
image1 = color.rgb2gray(image1)

mag1 = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(image1))))
mag = mag1/mag1.max()

io.imsave('outputs/image_intro.png', np.clip(image1*255, a_min=0, a_max=255.))
io.imsave('outputs/mag_intro.png', np.clip(mag*255, a_min=0, a_max=255.))
