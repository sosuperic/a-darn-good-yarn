# Analyze outputs, e.g. cluster shapes

# TODO: calculate k-means score
# TODO: what to do about shorts, should we filter by length, idk

import argparse
import hdbscan
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import os
import pandas as pd
import cPickle as pickle
import scipy.interpolate as interp
import sqlite3

# from core.predictions.spatio_time_cluster import *
from core.predictions.dtw_kmeans import *
from core.predictions.hierarchical_cluster import *
from core.predictions.utils import DTWDistance, fastdtw_dist, LB_Keogh, smooth
from core.utils.utils import setup_logging, get_credits_idx

# For local vs shannon`
PRED_FN = 'sent_biclass.csv' if os.path.abspath('.').startswith('/Users/eric') else 'sent_biclass_19.csv'

OUTPUTS_PATH = 'outputs/cluster/'
VIDEOS_PATH = 'data/videos'

TS_STR = 'dir{}-n{}-w{}-ds{}-maxnf{}-fn{}'
HDBSCAN_STR = 'dir{}-n{}-w{}-ds{}-maxnf{}-fn{}-r{}'
KMEANS_STR = 'dir{}-n{}-w{}-ds{}-maxnf{}-fn{}-k{}-it{}-r{}'

VIDEOPATH_DB = 'data/db/VideoPath.db'
VIDEOMETADATA_DB = 'data/db/VideoMetadata.pkl'

class Analysis(object):
    def __init__(self):
        self.logger = self._get_logger()

    ####################################################################################################################
    # Preprocess data
    ####################################################################################################################
    def prepare_and_save_ts(self, vids_dirpath, w, ds, max_nframes, pred_fn):
        """
        Create np array of [num_timeseries, max_len]

        Parameters
        ----------
        vids_dirpath:
        w: int, window size to use for smoothing
        pred_fn: filename that contains predictions, e.g. sent_biclass_19.csv
        ds: int, ratio at which to downsample time series
            - used to speed up clustering
            - e.g. 3 = sample every third point
        """

        # Get all series from videos with predictions
        self.logger.info('Preparing timeseries')
        ts = []
        ts_idx2title = {}

        vids_dirpaths = self.get_all_vidpaths_with_frames_and_preds(vids_dirpath)
        i = 0

        # For every video, get smoothed and downsampled time series
        self.logger.info('Getting predictions, removing credits preds, smoothing and downsampling')
        for vdp in vids_dirpaths:
            # self.logger.info('Video: {}'.format(vdp)

            # Get predictions
            preds_path = os.path.join(vdp, 'preds', pred_fn)
            vals = pd.read_csv(preds_path).pos.values

            # Skip if predictions are empty (some cases of this in shorts... I think because frames/ is empty)
            if len(vals) == 0:
                self.logger.info(u'{} predictions is 0, skipping'.format(unicode(vdp, 'utf-8')))
                continue

            # Remove predictions for credits frame
            credits_idx = get_credits_idx(vdp)
            if credits_idx:
                vals = vals[:credits_idx]

            # Skip if window size too big
            if len(vals) <= w:
                self.logger.info(u'{} length is {}, less than: {}, skipping'.format(
                    unicode(vdp, 'utf-8'), len(vals), w))  # unicode for titles
                continue

            # Skip if shorts too long (Only 7 are longer than 30 min: 30.03, 36.00, 41.12, 48.00, 49.60, 53.50))
            if len(vals) > max_nframes:
                self.logger.info(u'{} length is {}, greater than max_nframes ({})'.format(
                    unicode(vdp, 'utf-8'), len(vals), max_nframes))  # unicode for titles
                continue

            # Smooth and downsample
            smoothed = smooth(vals, window_len=w)
            downsampled = smoothed[::ds]

            ts.append(downsampled)
            ts_idx2title[i] = os.path.basename(vdp)
            i += 1

            # For debugging - to have quick results
            # if i == 5:
            #     break

        # Make all timeseries the same length by going from 0% of video to 100% of video and interpolating in between
        self.logger.info('Interpolating series to maximum series length')
        max_len = max([len(s) for s in ts])
        for i, s in enumerate(ts):
            if len(s) == max_len:
                continue
            else:
                s_interp = interp.interp1d(np.arange(s.size), s)
                s_stretch = s_interp(np.linspace(0, s.size-1, max_len))
                ts[i] = s_stretch

        # Convert into numpy
        ts = np.array(ts)  # (num_timeseries, max_len)

        # Normalize each series by taking mean and std of that one series
        self.logger.info('Z-normalizing each time series')
        mean = np.expand_dims(np.mean(ts, axis=1), 1)      # (num_timeseries, 1)
        std = np.expand_dims(ts.std(axis=1), 1)         # (num_timeseries, 1)
        ts = (ts - mean) / std


        # Save time series data
        # Save mean and std so we can map back to 0-1 later
        self.logger.info('Saving time series data')
        self._save_ts(ts, vids_dirpath, len(ts), w, ds, max_nframes, pred_fn)
        self._save_ts_idx2title(ts_idx2title, vids_dirpath, len(ts), w, ds, max_nframes, pred_fn)
        self._save_ts_mean(mean, vids_dirpath, len(ts), w, ds, max_nframes, pred_fn)
        self._save_ts_std(std, vids_dirpath, len(ts), w, ds, max_nframes, pred_fn)

        self.logger.info('Number of time series: {}'.format(len(ts)))
        self.logger.info('Time series length (max): {}'.format(ts.shape[1]))

        return ts


    ####################################################################################################################
    # Cluster
    ####################################################################################################################
    def cluster_ts(self, data, method, r, k=None, it=None):
        """
        Cluster data and save outputs

        Parameters
        ----------
        data: np of dimension [num_timeseries, max_len]
        method: str, clustering method to use
        k: comma-separated number of clusters (for parametric clustering techniques)
        """

        if method == 'kmeans':
            for k in k.split(','):
                print '=' * 100
                self.cluster_ts_kmeans(data, int(k), it, r)
        elif method == 'hierarchical':
            self.cluster_ts_hierarchical(data)
        elif method == 'hdbscan':
            self.cluster_ts_hdbscan(data, r)
        else:
            self.logger.info('Method unknown: {}'.format(method))

        self.logger.info('Done clustering')

    def cluster_ts_kmeans(self, data, k, it, r):
        """
        Parameters
        ----------
        data: np of dimension [num_timeseries, max_len]
        method: str, clustering method to use
        k: number of clusters (for parametric clustering techniques)
        """
        # Cluster
        self.logger.info('K-means clustering: k={}, it={}, r={}'.format(k, it, r))
        clusterer = ts_cluster(num_clust=k)
        clusterer.k_means_clust(data, it, 2, r)     # num_iter, w, r

        # Un-normalize so that saved outputs -- centroids are of sentiment in range 0,1
        mean = self.load_ts_mean(self.vids_dirpath, self.n, self.w, self.ds, self.max_nframes, self.pred_fn)
        std = self.load_ts_std(self.vids_dirpath, self.n, self.w, self.ds, self.max_nframes, self.pred_fn)
        for i, c in enumerate(clusterer.centroids):
            clusterer.centroids[i] = (c * std) + mean

        # Save outputs
        self.logger.info('Saving centroids, assignments, figure')
        self._save_kmeans(clusterer, k, it, r, self.vids_dirpath, self.n, self.w, self.ds, self.max_nframes, self.pred_fn)

    def cluster_ts_hierarchical(self, data):
        """
        Parameters
        ----------
        data: np of dimension [num_timeseries, max_len]
        """
        clusterer = HierarchicalCluster()
        clusterer.cluster(data)

    def cluster_ts_hdbscan(self, data, r):
        """

        Options
        -------
        1) Directly computing on [num_samples, timeseries]
        2) Compute distance matrix using LB_Keogh to create [num_samples,
        """
        # for d in LB_Keogh()
        self.logger.info('Clustering using HDBSCAN')

        self.logger.info('Creating DTW-based distance matrix')
        dist_matrix = np.zeros([len(data), len(data)])
        for i in range(len(data)):
            for j in range(i+1, len(data)):       # only calculate upper triangle
                # dist_matrix[i][j] = fastdtw_dist(data[i], data[j])
                dist_matrix[i][j] = LB_Keogh(data[i], data[j], 500) if i != j else 0.0
                # dist_matrix[i][j] = DTWDistance(data[i], data[j], 5) if i != j else 0.0
                print i, j, dist_matrix[i][j]

        self.logger.info('Saving DTW-based distance matrix')
        self._save_dtw_dist_matrix(dist_matrix, r, self.vids_dirpath, self.n, self.w, self.ds, self.max_nframes, self.pred_fn)

        self.logger.info('Clustering')
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
        clusterer.fit(dist_matrix)

        # Some logs
        self.logger.info('Number of clusters: {}'.format(clusterer.cluster_persistence_.shape[0]))
        self.logger.info('Cluster persistence: {}'.format(clusterer.cluster_persistence_))

        # Save
        self.logger.info('Saving clusterer')
        self._save_hdbscan(clusterer, r, self.vids_dirpath, self.n, self.w, self.ds, self.max_nframes, self.pred_fn)

        # Robust sinkle linkage (hierarchical?)
        # import hdbscan
        #
        clusterer = hdbscan.RobustSingleLinkage(cut=0.125, k=7)
        cluster_labels = clusterer.fit_predict(data)
        hierarchy = clusterer.cluster_hierarchy_
        alt_labels = hierarchy.get_clusters(0.100, 5)
        hierarchy.plot()
        plt.savefig('tmp.png')

        # for lab in cluster_labels:
        #     print cluster_labels
        # #
        # clusterer = HierarchicalCluster()
        # clusterer.cluster(data)



    ####################################################################################################################
    # Compute and save distances
    ####################################################################################################################
    # TODO: this should be a part of ts_cluster (distances are calculated anyway, return on last iteration)
    def compute_and_save_kmeans_clusters_ts_dists(self, k, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        """
        Compute and save distances between each time series and its centroid for k means clusters

        Parameters
        ----------
        - k: comma-separated list
        - Used to create file name and load saved outputs

        Saves
        ------
        ts_dists: dict, key is int (centroid_idx), value = dict (key is member_idx, value is distance)

        Notes
        -----
        Used to sort members by how close they are to the centroid for Clusters view in UGI
        """
        self.logger.info('Creating and saving distances')

        for k in k.split(','):
            print '=' * 100
            self.logger.info('k: {}'.format(k))
            k = int(k)

            try:
                ts = self.load_ts(vids_dirpath, n, w, ds, max_nframes, pred_fn)

                centroids = self.load_centroids(vids_dirpath, n, k, w, ds, max_nframes, pred_fn)
                assignments = self.load_assignments(vids_dirpath, n, k, w, ds, max_nframes, pred_fn)

                # Compute
                ts_dists = {}
                for c_idx, members in assignments.items():
                    ts_dists[c_idx] = {}
                    i = 0
                    for m_idx in members:
                        # ts_dists[c_idx][m_idx] =
                        ts_dists[c_idx][m_idx] = DTWDistance(centroids[c_idx],
                                                             ts[m_idx],
                                                             w=1)  # TODO: this w should be same as one used to cluster originally
                        if i  % 100 == 0:
                            self.logger.info('{} distances computed'.format(i))

                        i += 1

                # Save
                with open(os.path.join(OUTPUTS_PATH, 'data', 'ts_dists_{}.pkl'.format(clusters_str)), 'w') as f:
                    pickle.dump(ts_dists, f, protocol=2)

            except Exception as e:
                self.logger.info(e)

    ####################################################################################################################
    # Analyze 'groups' (combinations of different metadata, e.g. genre + year)
    ####################################################################################################################
    def save_and_analyze_groups(self):

        def get_groups():
            conn = sqlite3.connect(VIDEOPATH_DB)
            with conn:
                cur = conn.cursor()
                rows = cur.execute("SELECT title FROM VideoPath WHERE category=='films'")
                films = [row[0] for row in rows]

            video2metadata = pickle.load(open(VIDEOMETADATA_DB, 'r'))
            for film in films:
                metadata = video2metadata[film]

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
            if ('preds' in dirs) and ('frames' in dirs) \
                    and (PRED_FN in os.listdir(os.path.join(root, 'preds'))):
                vidpaths.append(root)

        return vidpaths

    def _save_ts(self, ts, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        params_str = self._get_TS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        ts_path = self._get_ts_path(params_str)
        with open(ts_path, 'wb') as f:
            pickle.dump(ts, f, protocol=2)

    def _save_ts_mean(self, mean, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        params_str = self._get_TS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        ts_mean_path = self._get_ts_mean_path(params_str)
        with open(ts_mean_path, 'wb') as f:
            pickle.dump(mean, f, protocol=2)

    def _save_ts_std(self, std, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        params_str = self._get_TS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        ts_std_path = self._get_ts_std_path(params_str)
        with open(ts_std_path, 'wb') as f:
            pickle.dump(std, f, protocol=2)

    def _save_ts_idx2title(self, ts_idx2title, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        params_str = self._get_TS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        ts_idx2title_path = self._get_ts_idx2title_path(params_str)
        with open(ts_idx2title_path, 'w') as f:
            pickle.dump(ts_idx2title, f, protocol=2)

    def load_ts(self, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        """
        Load timeseries (np array of dim [num_timeseries, max_len]) saved by prepare_and_save_timeseries(). Set 
        self fields because this method is called before clustering. Once clustering is done, will need these fields
        to save output.
        """
        self.logger.info('Loading ts')
        self.vids_dirpath = vids_dirpath
        self.n = n
        self.w = w
        self.ds = ds
        self.max_nframes = max_nframes
        self.pred_fn = pred_fn
        
        str = self._get_TS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        path = self._get_ts_path(str)
        ts = pickle.load(open(path, 'rb'))
        return ts

    def load_ts_mean(self, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        params_str = self._get_TS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        path = self._get_ts_mean_path(params_str)
        mean = pickle.load(open(path, 'rb'))
        return mean

    def load_ts_std(self, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        params_str = self._get_TS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        path = self._get_ts_std_path(params_str)
        std = pickle.load(open(path, 'rb'))
        return std

    def _save_kmeans(self, clusterer, k, it, r, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        """
        Save centroids, assignments, and plots for kmeans
        """
        params_str = self._get_KMEANS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r)
        centroids_path = self._get_centroids_path(params_str)
        assignments_path = self._get_assignments_path(params_str)
        with open(centroids_path, 'wb') as f:
            pickle.dump(clusterer.centroids, f, protocol=2)
        with open(assignments_path, 'wb') as f:
            pickle.dump(clusterer.assignments, f, protocol=2)

        # Plots
        for i, c in enumerate(clusterer.centroids):
            plt.plot(c, label=i)
        plt.legend()
        plt.savefig(os.path.join(OUTPUTS_PATH, 'imgs', 'kmeans_{}.png'.format(params_str)))
        plt.gcf().clear()               # clear figure so for next k

        # Some extra logging
        for centroid_idx, assignments in clusterer.assignments.items():
            self.logger.info('Centroid {}: {} series'.format(centroid_idx, len(assignments)))

    def _save_hdbscan(self, clusterer, r, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        """
        Save clusterer for hdbscan
        """
        params_str = self._get_HDBSCAN_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn, r)
        path = self._get_hdbscan_path(params_str)
        with open(path, 'wb') as f:
            pickle.dump(clusterer, f, protocol=2)

    def _save_dtw_dist_matrix(self, dtw_dist_matrix, r, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        params_str = self._get_HDBSCAN_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn, r)
        path = self._get_dtw_dist_matrix_path(params_str)
        with open(path, 'wb') as f:
            pickle.dump(dtw_dist_matrix, f, protocol=2)

    def load_centroids(self, k, it, r, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        """
        Load centroids from k means clustering
        """
        clusters_str = self._get_KMEANS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r)
        centroids_path = self._get_centroids_path(clusters_str)
        centroids = pickle.load(open(centroids_path, 'rb'))
        return centroids

    def load_assignments(self, k, it, r, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        """
        Load assignments from k means clustering
        """
        clusters_str = self._get_KMEANS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r)
        assignments_path = self._get_assignments_path(clusters_str)
        assignments = pickle.load(open(assignments_path, 'rb'))
        return assignments

    def _get_TS_STR_formatted(self, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        str = TS_STR.format(
            os.path.basename(vids_dirpath), n,
            w, ds, max_nframes,
            pred_fn[:-4])      # remove .csv
        return str

    def _get_KMEANS_STR_formatted(self, vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r):
        str = KMEANS_STR.format(
            os.path.basename(vids_dirpath), n, w,
            ds, max_nframes, pred_fn[:-4],
            k, it, r)      # remove .csv
        return str

    def _get_HDBSCAN_STR_formatted(self, vids_dirpath, n, w, ds, max_nframes, pred_fn, r):
        str = KMEANS_STR.format(
            os.path.basename(vids_dirpath), n, w,
            ds, max_nframes, pred_fn[:-4], r)      # remove .csv
        return str

    def _get_ts_path(self, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', 'ts_{}.pkl'.format(params_str))
        return path

    def _get_ts_mean_path(self, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', 'ts-mean_{}.pkl'.format(params_str))
        return path

    def _get_ts_std_path(self, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', 'ts-std_{}.pkl'.format(params_str))
        return path

    def _get_ts_idx2title_path(self, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', 'ts-idx2title_{}.pkl'.format(params_str))
        return path

    def _get_centroids_path(self, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', 'centroids_{}.pkl'.format(params_str))
        return path

    def _get_assignments_path(self, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', 'assignments_{}.pkl'.format(params_str))
        return path

    def _get_hdbscan_path(self, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', '{}.pkl'.format(params_str))
        return path

    def _get_dtw_dist_matrix_path(self, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', 'dtw-dist-matrix_{}.pkl'.format(params_str))
        return path

    def _get_logger(self):
        """Return logger, where path is dependent on mode (train/test), arch, and obj"""
        logs_path = os.path.join(os.path.dirname(__file__), 'logs')
        _, logger = setup_logging(save_path=os.path.join(logs_path, 'analysis.log'))
        return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster time series')

    # Action to take
    parser.add_argument('--prepare_and_save_ts', dest='prepare_and_save_ts', action='store_true', default=False)
    parser.add_argument('--cluster_ts', dest='cluster_ts', action='store_true', default=False)
    parser.add_argument('-m', '--method', dest='method', default='dtw', help='kmeans,hierarchical,hdbscan')
    parser.add_argument('--compute_and_save_kmeans_clusters_ts_dists', dest='compute_and_save_kmeans_clusters_ts_dists', action='store_true', default=False)

    # Parameters
    parser.add_argument('--vids_dirpath', dest='vids_dirpath', default=None,
                        help='folder to traverse for predictions, e.g. data/videos/films')
    parser.add_argument('-n', dest='n', default=None, help='n - get from filename')
    parser.add_argument('-w', dest='w', type=int, default=None, help='window size for smoothing predictions')
    parser.add_argument('-ds', dest='ds', type=int, default=None, help='downsample rate')
    parser.add_argument('--max_nframes', dest='max_nframes', type=int, default=float('inf'),
                        help='filter out videos with more frames than this. May be used with shorts to filter out'
                             'the high end (7 out of 1400 shorts are longer than 30 minutes')
    parser.add_argument('--pred_fn', dest='pred_fn', default=PRED_FN, help='pred file name')


    # Clustering-specific parameters
    parser.add_argument('-k', dest='k', default=None, help='list of comma-separated k to evaluate')
    parser.add_argument('-it', dest='it', type=int, default=10, help='Number of iterations for k-means')
    parser.add_argument('-r', dest='r', type=int, default=5, help='LB_Keogh window size')

    # Compute and save distances

    cmdline = parser.parse_args()

    analysis = Analysis()
    if cmdline.prepare_and_save_ts:
        analysis.prepare_and_save_ts(cmdline.vids_dirpath, cmdline.w,
                                             cmdline.ds, cmdline.max_nframes, cmdline.pred_fn)
    elif cmdline.cluster_ts:
        ts = analysis.load_ts(cmdline.vids_dirpath, cmdline.n, cmdline.w,
                              cmdline.ds, cmdline.max_nframes, cmdline.pred_fn)
        analysis.cluster_ts(ts, cmdline.method, cmdline.r, cmdline.k, cmdline.it)
    elif cmdline.compute_and_save_distances:
        analysis.compute_and_save_kmeans_clusters_ts_dists(cmdline.k, cmdline.it, cmdline.r,
                                                           cmdline.vids_dirpath, cmdline.n,
                                                           cmdline.w, cmdline.ds,
                                                           cmdline.max_nframes, cmdline.pred_fn)