import matplotlib.pylab as plt
import numpy as np
import random

from utils import DTWDistance, fastdtw_dist, LB_Keogh

class ts_cluster(object):
    def __init__(self, num_clust):
        """
        num_clust is the number of clusters for the k-means algorithm
        assignments holds the assignments of data points (indices) to clusters
        centroids holds the centroids of the clusters
        """
        self.num_clust = num_clust
        self.assignments = {}
        self.centroids = []

    def k_means_clust(self, data, num_iter, w, r, verbose=True):
        """
        k-means clustering algorithm for time series data.  dynamic time warping Euclidean distance
         used as default similarity measure.
        """
        self.centroids = random.sample(data, self.num_clust)

        for n in range(num_iter):
            if verbose:
                print 'iteration ' + str(n+1)

            # Assign data points to clusters
            self.assignments = {}
            for ind, i in enumerate(data):
                min_dist = float('inf')
                closest_clust = None
                for c_ind, j in enumerate(self.centroids):
                    # Attempt using fastdtw - fastdtw still much slower than LB_Keogh with r=5, but much faster than DTWDistance
                    # fastdtw is about 2 times slower than LB_Keogh with r=500, where ts length for films is ~15000
                    # cur_dist = fastdtw_dist(i, j)
                    # if cur_dist < min_dist:
                    #     min_dist = cur_dist
                    #     closest_clust = c_ind

                    # Attempt to only use LB_Keogh
                    cur_dist = LB_Keogh(i, j, r)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind

                    # Attempt to use LB_Keogh AND DTWDistance
                    # if LB_Keogh(i,j,5)<min_dist:
                    #     cur_dist=DTWDistance(i,j,w)
                    #     if cur_dist<min_dist:
                    #         min_dist=cur_dist
                    #         closest_clust=c_ind
                if closest_clust in self.assignments:
                    self.assignments[closest_clust].append(ind)
                else:
                    self.assignments[closest_clust]=[]

            # Recalculate centroids of clusters
            for key in self.assignments:
                clust_sum = np.zeros(len(data[0]))
                for k in self.assignments[key]:
                    clust_sum = np.add(clust_sum,data[k])
                    self.centroids[key]= [m / len(self.assignments[key]) for m in clust_sum]

    def get_centroids(self):
        return self.centroids

    def get_assignments(self):
        return self.assignments

    def plot_centroids(self):
        for i in self.centroids:
            plt.plot(i)
        plt.show()
    