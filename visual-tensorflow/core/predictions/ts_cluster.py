import matplotlib.pylab as plt
import numpy as np
import random

from utils import DTWDistance

class ts_cluster(object):
    def __init__(self,num_clust):
        '''
        num_clust is the number of clusters for the k-means algorithm
        assignments holds the assignments of data points (indices) to clusters
        centroids holds the centroids of the clusters
        '''
        self.num_clust=num_clust
        self.assignments={}
        self.centroids=[]

    def k_means_clust(self,data,num_iter,w,progress=False):
        '''
        k-means clustering algorithm for time series data.  dynamic time warping Euclidean distance
         used as default similarity measure.
        '''
        self.centroids=random.sample(data,self.num_clust)

        for n in range(num_iter):
            if progress:
                print 'iteration '+str(n+1)
            #assign data points to clusters
            self.assignments={}
            for ind,i in enumerate(data):
                min_dist=float('inf')
                closest_clust=None
                for c_ind,j in enumerate(self.centroids):
                    if self.LB_Keogh(i,j,5)<min_dist:
                        cur_dist=DTWDistance(i,j,w)
                        if cur_dist<min_dist:
                            min_dist=cur_dist
                            closest_clust=c_ind
                if closest_clust in self.assignments:
                    self.assignments[closest_clust].append(ind)
                else:
                    self.assignments[closest_clust]=[]

            #recalculate centroids of clusters
            for key in self.assignments:
                clust_sum=np.zeros(len(data[0]))
                for k in self.assignments[key]:
                    clust_sum=np.add(clust_sum,data[k])
                    self.centroids[key]=[m/len(self.assignments[key]) for m in clust_sum]
            # for key in self.assignments:
            #     clust_sum=0
            #     for k in self.assignments[key]:
            #         clust_sum=clust_sum+data[k]
            #     self.centroids[key]=[m/len(self.assignments[key]) for m in clust_sum]

    def get_centroids(self):
        return self.centroids

    def get_assignments(self):
        return self.assignments

    def plot_centroids(self):
        for i in self.centroids:
            plt.plot(i)
        plt.show()

    def LB_Keogh(self,s1,s2,r):
        '''
        Calculates LB_Keough lower bound to dynamic time warping. Linear
        complexity compared to quadratic complexity of dtw.
        '''
        LB_sum=0
        for ind,i in enumerate(s1):

            lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
            upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

            if i>upper_bound:
                LB_sum=LB_sum+(i-upper_bound)**2
            elif i<lower_bound:
                LB_sum=LB_sum+(i-lower_bound)**2

        return np.sqrt(LB_sum)
    