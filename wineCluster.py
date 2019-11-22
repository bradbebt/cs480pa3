#    name: wineCluster.py
# authors: Ben Bradberry, Elena Trafton
# purpose: K-Means compression on the wine dataset

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import csv
from sklearn import datasets
import sklearn.cluster as cluster
from sklearn.cluster import KMeans
from math import log, e
import scipy
from sklearn.metrics import silhouette_score

import argparse

'''
a helper function to parse arguments.
'''


def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--k', action='store', dest='k', default="", type=int, required=True, help='number of clusters')

    return parser.parse_args()


'''
This is a simple helper method that gets the correct cluster index
given an array of centroids and the desired cluster's centroid.
'''


def get_cluster_index(centroid, centroids):
    for k in range(0, centroids.shape[0]):
        if np.array_equal(centroid, centroids[k]):
            return k

    return np.Inf


'''
returns one array where each element is all of the points of a particular cluster.
clusters[i] has centroids[i].
'''


def get_clusters(X, X_compressed, centroids):
    clusters = [None] * len(centroids)  # initialize to size equivalent len(centroids)
    clusters = np.asarray(clusters)
    for i in range(0, X_compressed.shape[0]):
        point = X[i]
        centroid = X_compressed[i]

        clusterIndex = get_cluster_index(centroid, centroids)
        pixel_cluster = clusters[clusterIndex]
        if np.array_equal(pixel_cluster, None):
            pixel_cluster = [point]
        else:
            pixel_cluster = np.concatenate((pixel_cluster, [point]))
        clusters[clusterIndex] = pixel_cluster
    return clusters


'''
the average distance from (x, y) to every point in pixel_cluster.
'''


def intra_cluster_distance(X, pixel_cluster, x):
    distances = []
    this_point = X[x]

    for point in pixel_cluster:
        distances.append(scipy.spatial.distance.euclidean(point, this_point))
    return np.average(distances)


'''
It may be more correct to compute the average distance of (x, y) to every point in each 
cluster and determine from that which is the nearest cluster. However, that seems 
inefficient when I could just compare to the centroid of each cluster, so that's what I'm 
doing.

This function returns the average distance from (x, y) to each point in the nearest
cluster to which (x, y) does NOT belong.
'''


def nearest_cluster_distance(X, X_compressed, centroids, clusters, x):
    this_point = X[x]
    this_centroid = X_compressed[x]

    # locate nearest centroid
    min_distance = np.Inf
    nearest_centroid = None
    for i in range(0, centroids.shape[0]):
        temp = centroids[i]
        distance = scipy.spatial.distance.euclidean(temp, this_centroid)
        if ((not np.array_equal(this_centroid, temp)) and distance < min_distance):
            min_distance = distance
            nearest_centroid = temp

    # using knowledge of nearest centroid, get all points in that cluster
    nearest_cluster_index = get_cluster_index(nearest_centroid, centroids)
    nearest_cluster = clusters[nearest_cluster_index]

    # get average from this_point to all points in nearest cluster
    distances = []
    for point in nearest_cluster:
        distances.append(scipy.spatial.distance.euclidean(point, this_point))

    return np.average(distances)


def main():
    parms = parseArguments()

    wineDataset = datasets.load_wine().data
    print("dataset shape:", wineDataset.shape)
    wineTargets = datasets.load_wine().target  # the categories/labels of each wine, can be 0, 1, or 2
    kClusters = parms.k
    kArr = []
    entArr = []
    silArr = []
    RSSArr = []

    # In final version, don't take in an argument but instead loop through with the values 2, 3, 4, 5, 6
    for z in range(2, kClusters + 1):

        wineKM = cluster.KMeans(init="random", n_init=1, n_clusters=z, verbose=0)
        wineKM.fit(wineDataset)

        centroids = wineKM.cluster_centers_.squeeze()
        labels = wineKM.labels_
        inertia = wineKM.inertia_
        silhouettes = []
        # ent = entropy(labels, wineTargets)
        # ent = entropy(wineTargets, labels, e)
        val, count = np.unique(labels, return_counts=True)
        ent = entropy(val, count)
        # print("K: ", i)
        # print("Inertia: ", inertia)
        # print("Silhouette: ", silhouette)
        # print("Label size: ", labels.shape)
        # print("Entropy: ", ent)

        kArr.append(z)
        entArr.append(ent)
        RSSArr.append(inertia)

        wine_compressed = np.empty((wineDataset.shape[0], wineDataset.shape[1]))
        for i in range(0, labels.shape[0]):
            wine_compressed[i] = centroids[labels[i]]

        # silhouette calulations
    #    First, get a list of clusters.
    #    Then, for each pixel, identify that pixel's centroid and cluster,
    #        and use those and the intra_cluster_distance() and nearest_cluster_distance()
    #        functions to get that pixel's silhouette coefficient, and add it to an array
    #        that keeps each pixel's silhouette coefficient.
    #    Finally, simply take the average of that array. this is the average silhouette
    #        coefficient for this compressed image.

        clusters = get_clusters(wineDataset, wine_compressed, centroids)
        silhouettes = []
        for i in range(0, wineDataset.shape[0]):
            silhouette = np.Inf
            centroid = wine_compressed[i]
            cluster_index = get_cluster_index(centroid, centroids)
            pixel_cluster = clusters[cluster_index]
            a = intra_cluster_distance(wineDataset, pixel_cluster, i)
            b = nearest_cluster_distance(wineDataset, wine_compressed, centroids, clusters, i)
            if a < b:
                silhouette = 1 - (a / b)
            elif a == b:
                silhouette = 0
            elif a > b:
                silhouette = (b / a) - 1
            silhouettes.append(silhouette)

        print("average silhouette score:", np.average(silhouettes))
        silArr.append(np.average(silhouettes))

    # Generating Plots
    # Plot 1, RSS on Y, K on X
    plt.subplot(1, 3, 1)
    plt.scatter(kArr, RSSArr, c="blue")
    plt.title("RSS to K")
    plt.xlabel("K")
    plt.ylabel("RSS")

    # Plot 2, Silhouette on Y, K on X
    plt.subplot(1, 3, 2)
    plt.scatter(kArr, silArr, c="red")
    plt.title("Silhouette to K")
    plt.xlabel("K")
    plt.ylabel("Silhouette")

    # Plot 3, inertia on Y, K on X
    plt.subplot(1, 3, 3)
    plt.scatter(kArr, entArr, c="green")

    plt.title("Entropy to K")
    plt.xlabel("K")
    plt.ylabel("Entropy")

    plt.tight_layout()
    plt.show()
    plt.savefig("WinePlots.png")


if __name__ == '__main__':
    main()
