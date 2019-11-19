#    name: imgCompKMeans.py
#  author: molloykp (Nov 2019)
# purpose: K-Means compression on an image

import numpy as np

from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans as kmeans
from sklearn.preprocessing import StandardScaler

import argparse


def parseArguments():
    parser = argparse.ArgumentParser(
        description='KMeans compression of images')

    parser.add_argument('--imageFileName', action='store',
                        dest='imageFileName', default="", required=True,
                        help='input image file')
    parser.add_argument('--k', action='store',
                        dest='k', default="", type=int, required=True,
                        help='number of clusters')

    parser.add_argument('--outputFileName', action='store',
                       dest='outputFileName', default="", required=True,
                       help='output imagefile name')

    return parser.parse_args()

def main():
    parms = parseArguments()

    img = imread(parms.imageFileName)
    kTimes = parms.k
    img_size = img.shape

    # Reshape it to be 2-dimension
    # in other words, its a 1d array of pixels with colors (RGB)

    X = img.reshape(img_size[0] * img_size[1], img_size[2])

    # Insert your code here to perform
    # -- KMeans clustering
    # -- replace colors in the image with their respective centroid

    # Init must be random and n_init must be 1
    kmeans.n_iter_ = kTimes # setting kmeans to iterate k times
    X_compressed = KMeans(init="random", n_init=1, n_clusters=15, verbose=1).fit(X) # use loop to run 10 times?

    # Document Instructions:
    #
    # For one of the images that has been supplied, run kmeans 10 times with k = 15 and
    # report/plot the sum of the squared errors (inertia_).

    # Briefly explain why the results vary (1-2 sentences).


    # save modified image (code assumes new image in a variable
    # called X_compressed)
    # Reshape to have the same dimension as the original image 

    X_compressed.reshape(img_size[0], img_size[1], img_size[2])

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    ax.imshow(X_compressed)
    for ax in fig.axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(parms.outputFileName,dpi=400,bbox_inches='tight',pad_inches=0.05)
    plt.show()

if __name__ == '__main__':
    main()

