# Analyze outputs, e.g. cluster shapes

import os
import pandas as pd
import scipy.interpolate as interp

# from core.predictions.spatio_time_cluster import *
from core.predictions.ts_cluster import *
from core.predictions.utils import smooth

class Analysis(object):
    def __init__(self):
        pass

    def cluster_timeseries(self, data):
        clusterer = ts_cluster(num_clust=2)
        clusterer.k_means_clust(data, 1, 2)     # num_iter, window_size

        print clusterer.assignments
        # print clusterer.centroids

        clusterer.plot_centroids()

    def prepare_timeseries(self):
        """Create np array of [num_timeseries, max_len]"""

        # Get all series from videos with predictions
        series = []
        vidpaths = self.get_all_vidpaths_with_frames('data/videos')
        for vp in vidpaths:
            preds_path = os.path.join(vp, 'preds', 'sent_biclass.csv')
            series.append(smooth(pd.read_csv(preds_path).pos.values, window_len=120))

        # Make all timeseries the same length by going from 0% of video to 100% of video and interpolating in between
        max_len = max([len(s) for s in series])
        for i, s in enumerate(series):
            if len(s) == max_len:
                continue
            else:
                s_interp = interp.interp1d(np.arange(s.size), s)
                s_stretch = s_interp(np.linspace(0, s.size-1, max_len))
                series[i] = s_stretch

        # TODO: interpolate each series value between min and max value so as not to include absolute value

        data = np.array(series)  # (num_timeseries, max_len)
        return data

    def get_all_vidpaths_with_frames(self, starting_dir):
        """
        Return list of full paths to every video directory that contains preds/
        e.g. [<VIDEOS_PATH>/@Animated/@OldDisney/Feast/, ...]
        """
        vidpaths = []
        for root, dirs, files in os.walk(starting_dir):
            if 'preds' in os.listdir(root):
                vidpaths.append(root)

        return vidpaths

if __name__ == '__main__':
    analysis = Analysis()
    data = analysis.prepare_timeseries()
    analysis.cluster_timeseries(data)