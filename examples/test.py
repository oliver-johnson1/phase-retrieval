import numpy as np
from imageio import imread
from skimage import color
import matplotlib.pyplot as plt
import skimage.io as io
from util.util import reconstructImage, padImage

def plotDigits(images_all,images_original,out_filename):
    """
    This function plots the MNIST data in an array
    Inputs:
    images_all: list of list of reconstructed images
    images_original: list of original images
    out_filename: filename of saved plot
    """
    labels = ['GS','HIO','OSS']
    N = len(images_all)
    f, axarr = plt.subplots(N,4)
    f.set_figheight(3)
    f.set_figwidth(3)
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for ind in range(0,N):
        axarr[ind,0].imshow(images_original[ind],cmap = 'gray')
        axarr[ind,0].xaxis.set_visible(False)
        axarr[ind,0].yaxis.set_visible(False)
        if (ind == N-1):
                plt.text(0.5, -0.2, 'Original', \
                horizontalalignment='center', verticalalignment='center', \
                transform=axarr[ind,0].transAxes)
        for j in range(0,3):
            axarr[ind,j+1].imshow(images_all[ind][j],cmap = 'gray')
            axarr[ind,j+1].xaxis.set_visible(False)
            axarr[ind,j+1].yaxis.set_visible(False)
            if (ind == N-1):
                plt.text(0.5, -0.2, labels[j], \
                horizontalalignment='center', verticalalignment='center', \
                transform=axarr[ind,j+1].transAxes)
    plt.savefig('outputs/' + out_filename ,dpi=300)

def runLargeImage(small = False, noise = 0, iterations = 10000):
    # Read images
    image = imread('data/BSDS300/108073_smaller.jpg') 
    image = color.rgb2gray(image)

    # small section
    if small:
        image = image[20:90,20:90]

    # Pad Image
    image_padded, padding_key = padImage(image)

    # Compute FFT image
    mag = np.abs(np.fft.fft2(image_padded))

    # Add noise to FFT magnitudes
    mag = mag + np.random.normal(0,noise,mag.shape)

    # Initialize arrays
    image_data = [None]*3
    psnrs = [None]*3

    # GE
    image_data[0], psnrs[0] = reconstructImage(image, mag,padding_key,'gs',steps=iterations)
    
    # Hybrid
    image_data[1], psnrs[1] = reconstructImage(image, mag,padding_key,'hio',steps=iterations)
    # OSS
    image_data[2], psnrs[2] = reconstructImage(image, mag,padding_key,'oss',steps=iterations)

    # Save images
    io.imsave('outputs/original' + str(noise) + '.png', np.clip(image*255, a_min=0, a_max=255.))
    io.imsave('outputs/gs_mask' + str(noise) + '.png', np.clip(image_data[0]*255, a_min=0, a_max=255.))
    io.imsave('outputs/hio_mask' + str(noise) + '.png', np.clip(image_data[1]*255, a_min=0, a_max=255.))
    io.imsave('outputs/oss_mask'+ str(noise) + '.png', np.clip(image_data[2]*255, a_min=0, a_max=255.))
    print('PSNR')
    print('GS:', psnrs[0], ' HIO:',psnrs[1],' OSS:', psnrs[2])
    return psnrs

def runMNIST():
    # Load sample MNIST files
    filenames = ['data/MNIST_training/2/213.png',\
                'data/MNIST_training/4/58.png',\
                'data/MNIST_training/5/219.png',\
                'data/MNIST_training/8/545.png']
    images_all=[None]*len(filenames)
    images_original = [None]*len(filenames)

    # Reconstruct images with padding
    for ind in range(0,len(filenames)):
        image = imread(filenames[ind])
        image = image/255.
        images_original[ind]=image
        image_padded, padding_key = padImage(image)
        mag = np.abs(np.fft.fft2(image_padded))

        image_data = [None]*3

        # GE
        image_data[0], _ = reconstructImage(image, mag,padding_key,'gs',steps=3000)
        # Hybrid
        image_data[1], _ = reconstructImage(image, mag,padding_key,'hio',steps=3000)
        # OSS
        image_data[2], _ = reconstructImage(image, mag,padding_key,'oss',steps=3000)

        images_all[ind]=image_data

    # Make 4 x 4 plot
    plotDigits(images_all, images_original,'digs_mask.png')

    # Reconstruct images with no padding
    for ind in range(0,len(filenames)):
        image = imread(filenames[ind])
        image = image/255.
        images_original[ind]=image
        mag = np.abs(np.fft.fft2(image))
        image_data = [None]*3
        padding_key = None

        # GE
        image_data[0], _ = reconstructImage(image, mag,padding_key,'gs',steps=3000)

        # Hybrid
        image_data[1], _ = reconstructImage(image, mag,padding_key,'hio',steps=3000)

        # OSS
        image_data[2], _ = reconstructImage(image, mag,padding_key,'oss',steps=3000)
        
        images_all[ind]=image_data

    plotDigits(images_all, images_original,'digs_nomask.png')


def main():
    
    # tests big image
    runLargeImage(iterations=10000)
    runLargeImage(iterations=20000)

    allPsnrs = []
    # tests noise
    for _ in range(10):
        allPsnrs.append(runLargeImage(small = True, noise = 1))

    print(allPsnrs)
    print(np.mean(np.array(allPsnrs),axis=0))

    # runs MNIST plots
    runMNIST()
  
  
# Using the special variable 
# __name__
if __name__=="__main__":
    main()