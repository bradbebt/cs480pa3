#    name: imgCompKMeans.py
#  author: molloykp (Nov 2019)
#          Elena Trafton and Ben Bradberry
# purpose: K-Means compression on an image

import numpy as np

from matplotlib.image import imread
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from sklearn.cluster import KMeans as kmeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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
    print("image size:", img_size)

    # Reshape it to be 2-dimension- in other words, it's a 1d array of pixels with colors (RGB)
    img_y = img_size[0]
    img_x = img_size[1]

    X = img.reshape(img_size[0] * img_size[1], img_size[2])
    # Insert your code here to perform
    # Init must be random and n_init must be 1
    #kmeans.n_iter_ = kTimes # setting kmeans to iterate k times
    # -- KMeans clustering
    postKM = cluster.KMeans(init="random", n_init=1, n_clusters=kClusters, verbose=0) # use loop to run 10 times?
    postKM.fit(X)
    # -- replace colors in the image with their respective centroid

    centroids = postKM.cluster_centers_.squeeze() # an array of centroids
    labels = postKM.labels_   # print shape, labels what cluster each pixel is with
    print("centroids:", centroids)

    labels = labels.reshape(img_size[0], img_size[1])

    # creating the compressed image
    X_compressed  = np.empty((img_size[0], img_size[1], 3))
    for i in range(0, labels.shape[0]):
        for j in range(0, labels.shape[1]):
            X_compressed[i][j] = centroids[labels[i][j]]


    fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    ax.imshow(X_compressed)

    for ax in fig.axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(parms.outputFileName,dpi=400,bbox_inches='tight',pad_inches=0.05)

    plt.show()
    
    #silhouette calulations
    #	First, get a list of clusters.
    #	Then, for each pixel, identify that pixel's centroid and cluster,
    #		and use those and the intra_cluster_distance() and nearest_cluster_distance() 
    #		functions to get that pixel's silhouette coefficient, and add it to an array
    #		that keeps each pixel's silhouette coefficient.
    #	Finally, simply take the average of that array. this is the average silhouette
    #		coefficient for this compressed image.
    X = X.reshape(img_size[0], img_size[1], img_size[2])
    clusters = get_clusters(X, X_compressed, centroids)
    silhouettes = []
    for i in range(0, X.shape[0]):
    	for j in range(0, X.shape[1]):
    		silhouette = np.Inf
    		centroid = X_compressed[i][j]
    		cluster_index = get_cluster_index(centroid, centroids)
    		pixel_cluster = clusters[cluster_index]
    		a = intra_cluster_distance(X, pixel_cluster, i, j)
    		b = nearest_cluster_distance(X, X_compressed, centroids, clusters, i, j)
    		if a < b:
    			silhouette = 1 - (a/b)
    		elif a == b:
    			silhouette = 0
    		elif a > b:
    			silhouette = (b/a)-1
    		silhouettes.append(silhouette)
		
    print("~~~sil", np.average(silhouettes))
    
'''
returns one array where each element is all of the points of a particular cluster.
clusters[i] has centroids[i].
'''
def get_clusters(X, X_compressed, centroids):
	clusters = [None]*len(centroids) #initialize to size equivalent len(centroids)
	clusters = np.asarray(clusters)
	for i in range(0, X_compressed.shape[0]):
		for j in range(0, X_compressed.shape[1]):
			point = X[i][j]
			centroid = X_compressed[i][j]
			
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
def intra_cluster_distance(X, pixel_cluster, x, y):
    distances = []
    this_point = X[x][y]

    for point in pixel_cluster:
        #if i == x and j == y:
        #    break #don't include self
        distances.append(np.sqrt(np.square(point[0]-this_point[0]) +
                                 np.square(point[1]-this_point[1]) +
                                 np.square(point[2]-this_point[2])))
    return np.average(distances)

'''
It may be more correct to compute the average distance of (x, y) to every point in each 
cluster and determine from that which is the nearest cluster. However, that seems 
inefficient when I could just compare to the centroid of each cluster, so that's what I'm 
doing.

This function returns the average distance from (x, y) to each point in the nearest
cluster to which (x, y) does NOT belong.
'''
def nearest_cluster_distance(X, X_compressed, centroids, clusters, x, y):
	
    this_point = X[x][y]
    this_centroid = X_compressed[x][y]
    
    #locate nearest centroid
    min_distance = np.Inf
    nearest_centroid = None
    for i in range(0, centroids.shape[0]):
            temp = centroids[i]
            distance = np.sqrt(np.square(temp[0]-this_centroid[0]) + np.square(temp[1]-this_centroid[1]) + np.square(temp[2]-this_centroid[2]))
            if ((not np.array_equal(this_centroid, temp)) and distance < min_distance):
                min_distance = distance
                nearest_centroid = temp
	
	#using knowledge of nearest centroid, get all points in that cluster
    nearest_cluster_index = get_cluster_index(nearest_centroid, centroids)
    nearest_cluster = clusters[nearest_cluster_index]
    
    #get average from this_point to all points in nearest cluster
    distances = []
    for point in nearest_cluster:
        distances.append(np.sqrt(np.square(point[0]-this_point[0]) +
                                 np.square(point[1]-this_point[1]) +
                                 np.square(point[2]-this_point[2])))

    return np.average(distances)
    
'''
This is a simple helper method that gets the correct cluster index
given an array of centroids and the desired cluster's centroid.
'''
def get_cluster_index(centroid, centroids):
		for k in range(0, centroids.shape[0]):
			if np.array_equal(centroid, centroids[k]):
				return k
		return np.Inf


if __name__ == '__main__':
    main()

