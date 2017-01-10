# Heirarchical cluster of time-series data
# Unlike dynamic time warping-based clustering, this method not designed specifically for time-series data
# However, do not have to choose k

import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt

class HierarchicalCluster(object):
    def __init__(self):
        pass

    def cluster(self, data):
        """data is np array of [num_timeseries, max_len]"""
        z = hac.linkage(data, 'single', 'correlation')

        # Plot the dendogram
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        hac.dendrogram(
            z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
        )
        plt.show()