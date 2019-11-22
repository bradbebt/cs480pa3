#    name: imgCompKMeans.py
#  author: molloykp (Nov 2019)
# purpose: K-Means compression on an image

import numpy as np

from matplotlib.image import imread
import matplotlib.pyplot as plt
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
    print("image size:", img_size)

    # Reshape it to be 2-dimension- in other words, it's a 1d array of pixels with colors (RGB)
    img_y = img_size[0]
    img_x = img_size[1]

    X = img.reshape(img_size[0] * img_size[1], img_size[2])
    print("X shape:", X.shape)
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
    print("labels shape:", labels.shape)
    print("centroids shape:", centroids.shape)

    # creating the compressed image
    X_compressed  = np.empty((img_size[0], img_size[1], 3))
    print("X_compressed shape:", X_compressed.shape)
    for i in range(0, labels.shape[0]):
        for j in range(0, labels.shape[1]):
            X_compressed[i][j] = centroids[labels[i][j]]

    # Document Instructions:
    #
    # For one of the images that has been supplied, run kmeans 10 times with k = 15 and
    # report/plot the sum of the squared errors (inertia_).

    # Briefly explain why the results vary (1-2 sentences).



    fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    ax.imshow(X_compressed)

    for ax in fig.axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(parms.outputFileName,dpi=400,bbox_inches='tight',pad_inches=0.05)

    plt.show()

    '''
    LEFT TODO:
    -silhouette coefficient
    -plots and writeup
    -choose last photo
    -wine dataset
    
    '''
    
    X = X.reshape(img_size[0], img_size[1], img_size[2])
    clusters = chunk(X, X_compressed, centroids)
    silhouttes = []
    for i in range(0, X.shape[0]):
    	if i % 100 == 0:
    		print("i=",i)
    	for j in range(0, X.shape[1]):
    		silhouette = np.Inf
    		centroid = X_compressed[i][j]
    		cluster_index = get_cluster_index(centroid, centroids)
    		pixel_cluster = clusters[cluster_index]
    		a = intra_cluster_distance(X, X_compressed, centroid, pixel_cluster, i, j)
    		b = nearest_cluster_distance(X, X_compressed, centroids, clusters, i, j)
    		if a < b:
    			silhouette = 1 - (a/b)
    		elif a == b:
    			silhouette = 0
    		elif a > b:
    			silhouette = (b/a)-1
    		silhouttes.append(silhouette)
    
    			
    print("~~~sil", np.average(silhouettes))
    
    #(b-a)/max(a,b) for a sample
    #mean silhouette coefficient over all samples
    
    #optimization: calculate each cluster and then feed into intra_cluster_distance with different params instead of calculating centroid and cluster each time
    
def get_cluster_index(centroid, centroids):
		for k in range(0, centroids.shape[0]):
			if np.array_equal(centroid, centroids[k]):
				return k
		return np.Inf
    
def chunk(X, X_compressed, centroids):
	clusters = [None]*len(centroids) #initialize to size equivalent len(centroids)
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
        #if i == x and j == y:
        #    break #don't include self
        if np.array_equal(X_compressed[i][j], centroid):
            distances.append(np.sqrt(np.square(point[0]-this_point[0]) +
                             np.square(point[1]-this_point[1]) +
                             np.square(point[2]-this_point[2])))
    return np.average(distances)

'''
I'm not sure how to decide which is the closest cluster; by computing the average point across each cluster and comparing to that, or comparing to the centroid, or to the nearest non-cluster single point/sample to thispoint. I'm just going to compare to centroids.
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

    nearest_cluster_index = centroids.index(nearest_centroid)
    nearest_cluster = clusters[nearest_cluster_index]
    
    #get average from this_point to all points in nearest cluster
    distances = []
    for point in nearest_cluster:
        distances.append(np.sqrt(np.square(point[0]-this_point[0]) +
                                 np.square(point[1]-this_point[1]) +
                                 np.square(point[2]-this_point[2])))

    return np.average(distances)


if __name__ == '__main__':
    main()

