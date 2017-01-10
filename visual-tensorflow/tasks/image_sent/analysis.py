# Analyze outputs, e.g. cluster shapes

import argparse
import os
import pandas as pd
import pickle
import scipy.interpolate as interp

# from core.predictions.spatio_time_cluster import *
from core.predictions.ts_cluster import *
from core.predictions.hierarchical_cluster import *
from core.predictions.utils import smooth
from core.utils.utils import setup_logging


class Analysis(object):
    def __init__(self, method):
        self.method = method
        self.logger = self._get_logger()

    def cluster_timeseries(self, data):
        """data is np array of [num_timeseries, max_len]"""
        if self.method == 'dtw':
            self.cluster_timeseries_dtw(data)
        elif self.method == 'hierarchical':
            self.cluster_timeseries_hierarchical(data)
        else:
            self.logger.info('Method unknown: {}'.format(self.method))

        self.logger.info('Done clustering')

    def cluster_timeseries_hierarchical(self, data):
        """data is np array of [num_timeseries, max_len]"""
        clusterer = HierarchicalCluster()
        clusterer.cluster(data)

    def cluster_timeseries_dtw(self, data):
        """data is np array of [num_timeseries, max_len]"""
        clusterer = ts_cluster(num_clust=4)
        # clusterer = ts_cluster(num_clust=10)

        self.logger.info('K-means clustering')
        clusterer.k_means_clust(data, 1, 2)     # num_iter, window_size

        print clusterer.assignments
        # print clusterer.centroids

        with open('centroids_tmp20-4.pkl', 'w') as f:
            pickle.dump(clusterer.centroids, f, protocol=2)

        with open('assignments_tmp20-4.pkl', 'w') as f:
            pickle.dump(clusterer.assignments, f, protocol=2)

        # clusterer.plot_centroids()

    def prepare_timeseries(self):
        """Create np array of [num_timeseries, max_len]"""

        # Get all series from videos with predictions
        self.logger.info('Getting prediction series')
        series = []
        vidpaths = self.get_all_vidpaths_with_frames('data/videos')
        # vidpaths = self.get_all_vidpaths_with_frames('data/videos/films')
        for vp in vidpaths:
            preds_path = os.path.join(vp, 'preds', 'sent_biclass.csv')
            vals = pd.read_csv(preds_path).pos.values
            window_len = 1000
            if len(vals) < window_len:
                self.logger.info('{} length less than: {}'.format(vp, window_len))
                continue
            smoothed = smooth(vals, window_len=window_len)
            downsampled = smoothed[::3]
            series.append(downsampled)

        series = series[0:20]

        self.logger.info('{} series'.format(len(series)))
        self.logger.info('Interpolating series to maximum series length')
        # Make all timeseries the same length by going from 0% of video to 100% of video and interpolating in between
        max_len = max([len(s) for s in series])
        for i, s in enumerate(series):
            if len(s) == max_len:
                continue
            else:
                s_interp = interp.interp1d(np.arange(s.size), s)
                s_stretch = s_interp(np.linspace(0, s.size-1, max_len))
                series[i] = s_stretch

        # Interpolate each series value between min and max value so as not to include magnitude
        # i.e. min value is 0, max value is 1
        self.logger.info('Normalizing series')
        for i, s in enumerate(series):
            series[i] = (series[i] - series[i].min()) / (series[i].max() - series[i].min())

        data = np.array(series)  # (num_timeseries, max_len)

        with open('tmp_ts_data.pkl', 'w') as f:
            pickle.dump(data, f, protocol=2)

        # Uncomment this for hierarchical cluustering
        # data = pickle.load(open('tmp_ts_data_all.pkl', 'r'))

        print data.shape

        return data

    # Helper functions
    def get_all_vidpaths_with_frames(self, starting_dir):
        """
        Return list of full paths to every video directory that contains preds/
        e.g. [<VIDEOS_PATH>/@Animated/@OldDisney/Feast/, ...]
        """
        self.logger.info('Getting all vidpaths with frames/')
        vidpaths = []
        for root, dirs, files in os.walk(starting_dir):
            if 'preds' in os.listdir(root):
                vidpaths.append(root)

        return vidpaths

    def _get_logger(self):
        """Return logger, where path is dependent on mode (train/test), arch, and obj"""
        logs_path = os.path.join(os.path.dirname(__file__), 'logs')
        _, logger = setup_logging(save_path=os.path.join(logs_path, 'analysis.log'))
        return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster time series')
    parser.add_argument('-m', '--method', dest='method', default='dtw', help='dtw,hierarchical')
    cmdline = parser.parse_args()

    analysis = Analysis(cmdline.method)
    data = analysis.prepare_timeseries()
    analysis.cluster_timeseries(data)