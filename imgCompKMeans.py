#    name: imgCompKMeans.py
#  author: molloykp (Nov 2019)
# purpose: K-Means compression on an image

import numpy as np

from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
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
    kClusters = parms.k
    img_size = img.shape
    print(img_size)
    # Reshape it to be 2-dimension
    # in other words, its a 1d array of pixels with colors (RGB)

    X = img.reshape(img_size[0] * img_size[1], img_size[2])

    # Insert your code here to perform
    # Init must be random and n_init must be 1
    #kmeans.n_iter_ = kTimes # setting kmeans to iterate k times
    # -- KMeans clustering
    X_compressed = cluster.KMeans(init="random", n_init=1, n_clusters=kClusters, verbose=1).fit(X) # use loop to run 10 times?
    X_compressed.fit(X)
    # -- replace colors in the image with their respective centroid


    cluster_centers = X_compressed.cluster_centers_
    cluster_labels = X_compressed.labels_

    # Assign each pixel to one of the clusters

    # Document Instructions:
    #
    # For one of the images that has been supplied, run kmeans 10 times with k = 15 and
    # report/plot the sum of the squared errors (inertia_).

    # Briefly explain why the results vary (1-2 sentences).


    # save modified image (code assumes new image in a variable
    # called X_compressed)
    # Reshape to have the same dimension as the original image
    plt.figure(figsize=(15, 8))  ## Temp?
    plt.imshow(cluster_centers[cluster_labels].reshape(img_size[0], img_size[1], img_size[2]))  ## Temp?
    plt.show()

# All commented out lines below need to be uncommented out eventually
    #X_compressed.reshape(img_size[0], img_size[1], img_size[2]) # This is an original line

    #fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    #ax.imshow(X_compressed)
    #for ax in fig.axes:
    #    ax.axis('off')
    #plt.tight_layout()
    #plt.savefig(parms.outputFileName,dpi=400,bbox_inches='tight',pad_inches=0.05)
    #plt.show()

if __name__ == '__main__':
    main()
