# Analyze outputs, e.g. cluster shapes

# TODO: calculate k-means score
# TODO: what to do about shorts, should we filter by length, idk

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import os
import pandas as pd
import pickle
import scipy.interpolate as interp

# from core.predictions.spatio_time_cluster import *
from core.predictions.ts_cluster import *
from core.predictions.hierarchical_cluster import *
from core.predictions.utils import smooth
from core.utils.utils import setup_logging, get_credits_idx

OUTPUTS_PATH = 'outputs/cluster/'
VIDEOS_PATH = 'data/videos'

class Analysis(object):
    def __init__(self):
        self.logger = self._get_logger()

    ####################################################################################################################
    # Cluster
    ####################################################################################################################
    def cluster_timeseries(self, data, method, k):
        """
        Cluster data and save outputs

        Parameters
        ----------
        data: np of dimension [num_timeseries, max_len]
        method: str, clustering method to use
        k: number of clusters (for parametric clustering techniques)
        """
        if method == 'dtw':
            self.cluster_timeseries_dtw(data, k)
        elif method == 'hierarchical':
            self.cluster_timeseries_hierarchical(data)
        else:
            self.logger.info('Method unknown: {}'.format(method))

        self.logger.info('Done clustering')

    def cluster_timeseries_hierarchical(self, data):
        """
        Parameters
        ----------
        data: np of dimension [num_timeseries, max_len]
        """
        clusterer = HierarchicalCluster()
        clusterer.cluster(data)

    def cluster_timeseries_dtw(self, data, k=4):
        """
        Parameters
        ----------
        data: np of dimension [num_timeseries, max_len]
        method: str, clustering method to use
        k: number of clusters (for parametric clustering techniques)
        """
        clusterer = ts_cluster(num_clust=k)

        self.logger.info('K-means clustering: k={}'.format(k))
        clusterer.k_means_clust(data, 1, 2)     # num_iter, window_size

        # print clusterer.assignments
        # print clusterer.centroids

        # Save outputs
        self.logger.info('Saving centroids, assignments, figure')
        params_str = 'dir{}-n{}-k{}-w{}-ds{}-maxnf{}-fn{}'.format(
            os.path.basename(self.root_videos_dirpath),
            self.n, k, self.window_size, self.downsample_rate,
            self.max_num_frames,
            self.pred_fn[:-4])      # remove .csv
        with open(os.path.join(OUTPUTS_PATH, 'data', 'centroids_{}.pkl'.format(params_str)), 'w') as f:
            pickle.dump(clusterer.centroids, f, protocol=2)
        with open(os.path.join(OUTPUTS_PATH, 'data', 'assignments_{}.pkl'.format(params_str)), 'w') as f:
            pickle.dump(clusterer.assignments, f, protocol=2)
        # Plots
        for i, c in enumerate(clusterer.centroids):
            plt.plot(c, label=i)
        plt.legend()
        plt.savefig(os.path.join(OUTPUTS_PATH, 'imgs', 'dtw_{}.png'.format(params_str)))
        plt.gcf().clear()               # clear figure so for next k

        for centroid_idx, assignments in clusterer.assignments.items():
            self.logger.info('Centroid {}: {} series'.format(centroid_idx, len(assignments)))

        # clusterer.plot_centroids()

    ####################################################################################################################
    # Preprocess data
    ####################################################################################################################
    def prepare_timeseries(self, root_videos_dirpath, window_size, pred_fn, downsample_rate, max_num_frames):
        """
        Create np array of [num_timeseries, max_len]

        Parameters
        ----------
        root_videos_dirpath:
        window_size: int, window size to use for smoothing
        pred_fn: filename that contains predictions, e.g. sent_biclass_19.csv
        downsample_rate: int, ratio at which to downsample time series
            - used to speed up clustering
            - e.g. 3 = sample every third point
        """

        # Get all series from videos with predictions
        self.logger.info('Preparing timeseries')
        ts = []
        ts_idx2title = {}

        vid_dirpaths = self.get_all_vidpaths_with_frames_and_preds(root_videos_dirpath)
        i = 0

        self.logger.info('Getting predictions, removing credits preds, smoothing and downsampling')

        for vid_dirpath in vid_dirpaths:
            # self.logger.info('Video: {}'.format(vid_dirpath))

            # Get predictions
            preds_path = os.path.join(vid_dirpath, 'preds', pred_fn)
            vals = pd.read_csv(preds_path).pos.values

            # Skip if predictions are empty (some cases of this in shorts... I think because frames/ is empty)
            if len(vals) == 0:
                self.logger.info(u'{} predictions is 0, skipping'.format(unicode(vid_dirpath, 'utf-8')))
                continue

            # Remove predictions for credits frame
            credits_idx = get_credits_idx(vid_dirpath)
            if credits_idx:
                # self.logger.info('Credits index: {}'.format(credits_idx))
                vals = vals[:credits_idx]

            # Skip if window size too big
            if len(vals) <= window_size:
                self.logger.info(u'{} length is {}, less than: {}, skipping'.format(
                    unicode(vid_dirpath, 'utf-8'), len(vals), window_size))  # unicode for titles
                continue

            # Skip if shorts too long (Only 7 are longer than 30 min: 30.03, 36.00, 41.12, 48.00, 49.60, 53.50))
            if len(vals) > max_num_frames:
                self.logger.info(u'{} length greater than max_num_frames ({})'.format(
                    unicode(vid_dirpath, 'utf-8'), max_num_frames))  # unicode for titles
                continue

            # Smooth and downsample
            smoothed = smooth(vals, window_len=window_size)
            downsampled = smoothed[::downsample_rate]

            ts.append(downsampled)
            ts_idx2title[i] = os.path.basename(vid_dirpath)
            i += 1

            # For debugging - to have quick results
            # if i == 5:
            #     break

        self.logger.info('Interpolating series to maximum series length')
        # Make all timeseries the same length by going from 0% of video to 100% of video and interpolating in between
        max_len = max([len(s) for s in ts])
        for i, s in enumerate(ts):
            if len(s) == max_len:
                continue
            else:
                s_interp = interp.interp1d(np.arange(s.size), s)
                s_stretch = s_interp(np.linspace(0, s.size-1, max_len))
                ts[i] = s_stretch

        # Interpolate each series value between min and max value so as not to include magnitude
        # i.e. min value is 0, max value is 1
        self.logger.info('Normalizing range of each series')
        for i, s in enumerate(ts):
            ts[i] = (ts[i] - ts[i].min()) / (ts[i].max() - ts[i].min())

        ts = np.array(ts)  # (num_timeseries, max_len)

        # Save time series data
        self.logger.info('Saving time series data')
        params_str = 'dir{}-n{}-w{}-ds{}-maxnf{}-fn{}'.format(
            os.path.basename(root_videos_dirpath),
            len(ts), window_size, downsample_rate, max_num_frames,
            pred_fn[:-4])      # remove .csv
        with open(os.path.join(OUTPUTS_PATH, 'data', 'ts_{}.pkl'.format(params_str)), 'w') as f:
            pickle.dump(ts, f, protocol=2)
        with open(os.path.join(OUTPUTS_PATH, 'data', 'ts_idx2title_{}.pkl'.format(params_str)), 'w') as f:
            pickle.dump(ts_idx2title, f, protocol=2)

        # Save parameters to use in output filename
        self.root_videos_dirpath = root_videos_dirpath
        self.n = len(ts)
        self.window_size = window_size
        self.pred_fn = pred_fn
        self.downsample_rate = downsample_rate
        self.max_num_frames = max_num_frames

        self.logger.info('Number of time series: {}'.format(len(ts)))
        self.logger.info('Time series length (max): {}'.format(ts.shape[1]))

        return ts

    ####################################################################################################################
    # Helper functions
    ####################################################################################################################
    def get_all_vidpaths_with_frames_and_preds(self, starting_dir):
        """
        Return list of paths ([<VIDEOS_PATH>/films/M-VAD_full_movies/21 Jump Street/, ...]) to every directory that
        contains frames/ and preds/

        Parameters
        ----------
        starting_dir: e.g. data/videos/films
        """
        self.logger.info('Traversing: {}'.format(starting_dir))
        self.logger.info('Finding all directories with frames/ and preds/ folders')
        vidpaths = []
        for root, dirs, files in os.walk(starting_dir):
            if ('preds' in os.listdir(root)) and ('frames' in os.listdir(root)):
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
    parser.add_argument('--root_videos_dir', dest='root_videos_dir', default=VIDEOS_PATH,
                        help='folder to traverse for predictions')
    parser.add_argument('-k', dest='k', default='4', help='list of comma-separated k to evaluate')
    parser.add_argument('-w', dest='window_size', type=int, default=1000, help='window size for smoothing predictions')
    parser.add_argument('--pred_fn', dest='pred_fn', default='sent_biclass_19.csv', help='pred file name')
    parser.add_argument('-ds', dest='downsample_rate', type=int, default=3, help='downsample rate')
    parser.add_argument('--max_num_frames', dest='max_num_frames', type=int, default=float('inf'),
                        help='filter out videos with more frames than this. May be used with shorts to filter out'
                             'the high end (7 out of 1400 shorts are longer than 30 minutes')
    cmdline = parser.parse_args()

    analysis = Analysis()
    ts = analysis.prepare_timeseries(cmdline.root_videos_dir, cmdline.window_size,
                                       cmdline.pred_fn, cmdline.downsample_rate, cmdline.max_num_frames)
    for k in cmdline.k.split(','):
        print '=' * 100
        analysis.cluster_timeseries(ts, cmdline.method, int(k))