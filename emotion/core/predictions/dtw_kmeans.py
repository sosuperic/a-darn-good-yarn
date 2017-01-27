from collections import defaultdict
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
        self.ts_dists = defaultdict(dict)

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

                # Add distance b/n series i and centroid to ts_dists
                self.ts_dists[closest_clust][ind] = min_dist

                if closest_clust in self.assignments:
                    self.assignments[closest_clust].append(ind)
                else:
                    self.assignments[closest_clust] = []        # TODO: shouldn't this be [ind]

            # Recalculate centroids of clusters
            for key in self.assignments:
                clust_sum = np.zeros(len(data[0]))
                for k in self.assignments[key]:
                    clust_sum = np.add(clust_sum,data[k])
                    self.centroids[key]= [m / len(self.assignments[key]) for m in clust_sum]

    def k_means_clust_modifcentroids(self, data, dist_matrix, num_iter, w, r, verbose=True):
        self.centroids_ts_idxs = random.sample(range(len(data)), self.num_clust)
        self.ts_dists = defaultdict(dict)

        for n in range(num_iter):
            if verbose:
                print 'iteration ' + str(n+1)
                print self.centroids_ts_idxs

            self.assignments = {}
            for ts_idx, ts_i in enumerate(data):
                min_dist = float('inf')
                closest_clust = None
                for c_ts_idx in self.centroids_ts_idxs:
                    pair = sorted([ts_idx, c_ts_idx])     # get upper triangle
                    cur_dist = dist_matrix[pair[0]][pair[1]]
                    # centroid_ts = data[c_ts_idx]
                    # cur_dist = LB_Keogh(ts_i, centroid_ts, r)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ts_idx

                self.ts_dists[closest_clust][ts_idx] = min_dist

                if closest_clust in self.assignments:
                    self.assignments[closest_clust].append(ts_idx)
                else:
                    self.assignments[closest_clust] = [ts_idx]

            # Recalculate 'centroids' of clusters
            # Minimum pair-wise
            self.centroids_ts_idxs = []
            for c_ts_idx, members_idxs in self.assignments.items():
                sum_pair_wise_dists = []
                for i in range(len(members_idxs)):
                    cur_idx_sum_pair_wise_dists = []    # list of distances b/n cur point and all other points
                    for j in range(len(members_idxs)):
                        cur_idx_sum_pair_wise_dists.append(dist_matrix[members_idxs[i]][members_idxs[j]])
                        cur_idx_sum_pair_wise_dists.append(dist_matrix[members_idxs[j]][members_idxs[i]])
                    cur_idx_sum_pair_wise_dists = sum(cur_idx_sum_pair_wise_dists) / len(members_idxs)
                    sum_pair_wise_dists.append(cur_idx_sum_pair_wise_dists)
                new_c_ts_idx = members_idxs[sum_pair_wise_dists.index(min(sum_pair_wise_dists))]
                print min(sum_pair_wise_dists)
                self.centroids_ts_idxs.append(new_c_ts_idx)
                self.assignments[new_c_ts_idx] = members_idxs

        self.centroids = np.vstack([data[i] for i in self.centroids_ts_idxs])


    def get_centroids(self):
        return self.centroids

    def get_assignments(self):
        return self.assignments

    def get_ts_dists(self):
        return self.ts_dists

    def plot_centroids(self):
        for i in self.centroids:
            plt.plot(i)
        plt.show()
    