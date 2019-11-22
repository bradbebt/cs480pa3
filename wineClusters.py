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

import argparse


def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--k', action='store',
            dest='k', default="", type=int, required=True,
            help='number of clusters')

    return parser.parse_args()
def get_cluster_index(centroid, centroids):
    for k in range(0, centroids.shape[0]):
        if np.array_equal(centroid, centroids[k]):
            return k
    return np.Inf

def chunk(X, X_compressed, centroids):
    clusters = [None] * len(centroids)  # initialize to size equivalent len(centroids)
    for i in range(0, X_compressed.shape[0]):
        for j in range(0, X_compressed.shape[1]):
            point = X[i][j]
            centroid = X_compressed[i][j]

            clusterIndex = get_cluster_index(centroid, centroids)

            pixel_cluster = clusters[clusterIndex]
            if np.array_equal(pixel_cluster, None):
                pixel_cluster = point
            else:
                np.append(pixel_cluster, point)
            clusters[clusterIndex] = pixel_cluster
    clusters = np.asarray(clusters)
    print("X COMPRESSED SHAPE:", X_compressed.shape[0])
    print("SHAPE OF CLUSTERS:", clusters.shape)
    print("CLUSTERS:", clusters)
    return clusters

def intra_cluster_distance(X, X_compressed, centroid, pixel_cluster, x, y):
    distances = []
    this_point = X[x][y]

    for point in pixel_cluster:
        # if i == x and j == y:
        #    break #don't include self
        if np.array_equal(X_compressed[x][y], centroid):
            distances.append(np.sqrt(np.square(point[0] - this_point[0]) +
                                     np.square(point[1] - this_point[1]) +
                                     np.square(point[2] - this_point[2])))
    return np.average(distances)

'''
I'm not sure how to decide which is the closest cluster; by computing the average point across each cluster and comparing to that, or comparing to the centroid, or to the nearest non-cluster single point/sample to thispoint. I'm just going to compare to centroids.
'''

def nearest_cluster_distance(X, X_compressed, centroids, clusters, x, y):
    this_point = X[x][y]
    this_centroid = X_compressed[x][y]

    # locate nearest centroid
    min_distance = np.Inf
    nearest_centroid = None
    for i in range(0, centroids.shape[0]):
        temp = centroids[i]
        distance = np.sqrt(
            np.square(temp[0] - this_centroid[0]) + np.square(temp[1] - this_centroid[1]) + np.square(
                temp[2] - this_centroid[2]))
        if ((not np.array_equal(this_centroid, temp)) and distance < min_distance):
            min_distance = distance
            nearest_centroid = temp

    nearest_cluster_index = centroids.index(nearest_centroid)
    nearest_cluster = clusters[nearest_cluster_index]

    # get average from this_point to all points in nearest cluster
    distances = []
    for point in nearest_cluster:
        distances.append(np.sqrt(np.square(point[0] - this_point[0]) +
                                 np.square(point[1] - this_point[1]) +
                                 np.square(point[2] - this_point[2])))

    return np.average(distances)

def main():
    parms = parseArguments()

    wineDataset = datasets.load_wine().data
    wineTargets = datasets.load_wine().target  # the categories of each wine, can be 0, 1, or 2
    kClusters = parms.k
    kArr = []
    entArr = []
    silArr = []
    RSSArr = []

    # In final version, don't take in an argument but instead loop through with the values 2, 3, 4, 5, 6
    for z in range(2,kClusters + 1):

        wineKM = cluster.KMeans(init="random", n_init=1, n_clusters=z, verbose=0)
        wineKM.fit(wineDataset)

        centroids = wineKM.cluster_centers_.squeeze() # An array of centroids
        labels = wineKM.labels_
        inertia = wineKM.inertia_
        silhouettes = []
        #ent = entropy(labels, wineTargets)
        #ent = entropy(wineTargets, labels, e)
        val, count = np.unique(labels, return_counts=True)
        ent = entropy(val, count)
        #print("K: ", i)
        #print("Inertia: ", inertia)
        #print("Silhouette: ", silhouette)
        #print("Label size: ", labels.shape)
        #print("Entropy: ", ent)

        kArr.append(z)
        entArr.append(ent)
        RSSArr.append(inertia)

        print(centroids.shape[0])
        print(labels.shape[1])
        wine_compressed = np.empty((wineDataset.shape[0], wineDataset.shape[1]))
        print("X_compressed shape:", wine_compressed.shape)
        for i in range(0, labels.shape[0]):
            for j in range(0, labels.shape[1]):
                print(j)
                wine_compressed[i][j] = centroids[labels[i][j]]

        # Calculating silhouettes
        clusters = chunk(wineDataset, wine_compressed, centroids)
        for i in range(0, wineDataset.shape[0]):
            if i % 100 == 0:
                print("i=", i)
            for j in range(0, wineDataset.shape[1]):
                silhouette = np.Inf
                centroid = wine_compressed[i][j]
                cluster_index = get_cluster_index(centroid, centroids)
                pixel_cluster = clusters[cluster_index]
                a = intra_cluster_distance(wineDataset, wine_compressed, centroid, pixel_cluster, i, j)
                b = nearest_cluster_distance(wineDataset, wine_compressed, centroids, clusters, i, j)
                if a < b:
                    silhouette = 1 - (a / b)
                elif a == b:
                    silhouette = 0
                elif a > b:
                    silhouette = (b / a) - 1
                silhouettes.append(silhouette)

        silArr.append(silhouette)

        print("~~~sil", np.average(silhouettes))

        # (b-a)/max(a,b) for a sample
        # mean silhouette coefficient over all samples

        # optimization: calculate each cluster and then feed into intra_cluster_distance with different params instead of calculating centroid and cluster each time

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
