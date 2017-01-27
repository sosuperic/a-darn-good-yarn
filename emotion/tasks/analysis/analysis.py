# Analyze outputs, e.g. cluster shapes

# TODO: calculate k-means score
# TODO: what to do about shorts, should we filter by length, idk

import argparse
from concurrent.futures import ProcessPoolExecutor, wait
from functools import partial
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
from core.predictions.ts_cluster import *
from core.predictions.hierarchical_cluster import *
from core.predictions.utils import DTWDistance, fastdtw_dist, LB_Keogh, smooth
from core.utils.utils import setup_logging, get_credits_idx

# For local vs shannon`
PRED_FN = 'sent_biclass.csv' if os.path.abspath('.').startswith('/Users/eric') else 'sent_biclass_19.csv'

OUTPUTS_PATH = 'outputs/cluster/'
VIDEOS_PATH = 'data/videos'

TS_STR = 'dir{}-n{}-w{}-ds{}-maxnf{}-fn{}'
KCLUST_STR = 'dir{}-n{}-w{}-ds{}-maxnf{}-fn{}-k{}-it{}-r{}'
DIST_MATRIX_STR = GROUP_COHERENCE_STR = 'dir{}-n{}-w{}-ds{}-maxnf{}-fn{}-r{}'
HDBSCAN_STR = 'dir{}-n{}-w{}-ds{}-maxnf{}-fn{}-r{}-mcs{}-ms{}'

VIDEOPATH_DB = 'data/db/VideoPath.db'
VIDEOMETADATA_DB = 'data/db/VideoMetadata.pkl'

class Analysis(object):
    def __init__(self):
        self.logger = self._get_logger()

    ####################################################################################################################
    # Preprocess data
    ####################################################################################################################
    def prepare_ts(self, vids_dirpath, w, ds, max_nframes, pred_fn):
        """
        Create and save np array of [num_timeseries, max_len]

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
            # self.logger.info(u'Video: {}'.format(unicode(vdp, 'utf-8')))

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
            # If w is not None, then use uniform w for all videos
            if (w is not None) and (len(vals) <= w):
                self.logger.info(u'{} length is {}, less than: {}, skipping'.format(
                    unicode(vdp, 'utf-8'), len(vals), w))  # unicode for titles
                continue

            # Skip if video too long (e.g. only 7 shorts are longer than 30 min: 30.03, 36.00, 41.12, 48.00, 49.60, 53.50))
            if len(vals) > max_nframes:
                self.logger.info(u'{} length is {}, greater than max_nframes ({})'.format(
                    unicode(vdp, 'utf-8'), len(vals),  max_nframes))  # unicode for titles
                continue

            # Smooth and downsample
            # If w is None, then use 0.07 * video length. Got this val bc using w ~= 500 for films, avg. film is ~ 7200
            cur_w = w if w else int(0.07 * len(vals))
            smoothed = smooth(vals, window_len=cur_w)
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
        mean = np.expand_dims(np.mean(ts, axis=1), 1)       # (num_timeseries, 1)
        std = np.expand_dims(ts.std(axis=1), 1)             # (num_timeseries, 1)
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
    def cluster_ts(self, data, method, r, k=None, it=None, mcs=None, ms=None):
        """
        Cluster data and save outputs

        Parameters
        ----------
        data: np of dimension [num_timeseries, max_len]
        method: str, clustering method to use
        k: comma-separated number of clusters (for parametric clustering techniques)
        mcs: int, minimum_cluster_size
        ms: int, min_samples
        """

        if method == 'kmeans':
            for k in k.split(','):
                print '=' * 100
                self.cluster_ts_kmeans(data, int(k), it, r)
        elif method == 'kmedoids':
            dist_matrix = self._try_load_dtw_dist_matrix(self.vids_dirpath, self.n, self.w, self.ds, self.max_nframes, self.pred_fn, r)
            for k in k.split(','):
                print '=' * 100
                self.cluster_ts_kmedoids(data, dist_matrix, int(k), it)
        elif method == 'hierarchical':
            self.cluster_ts_hierarchical(data)
        elif method == 'hdbscan':
            self.cluster_ts_hdbscan(data, r, mcs, ms)
        else:
            self.logger.info('Method unknown: {}'.format(method))

        self.logger.info('Done clustering')

    def cluster_ts_kmeans(self, data, k, it, r):
        """
        Parameters
        ----------
        data: np of dimension [num_timeseries, max_len]
        k: number of clusters (for parametric clustering techniques)
        it: number of iterations
        r: window size for KB_Keogh
        """
        # Cluster
        self.logger.info('K-means clustering: k={}, it={}, r={}'.format(k, it, r))
        clusterer = ts_cluster(num_clust=k)
        clusterer.k_means_clust(data, it, 2, r)     # num_iter, w, r

        # Un-normalize so that saved outputs -- centroids are of sentiment in range 0,1
        # Mean and std is per time series, i.e. one per video
        # Not sure if this is the 'right' way to do it, but just take the mean of the mean and std across all videos
        mean = self._load_ts_mean(self.vids_dirpath, self.n, self.w, self.ds, self.max_nframes, self.pred_fn)
        std = self._load_ts_std(self.vids_dirpath, self.n, self.w, self.ds, self.max_nframes, self.pred_fn)
        mean = mean.mean()
        std = std.mean()
        for i, c in enumerate(clusterer.centroids):
            clusterer.centroids[i] = list((np.array(c) * std) + mean)

        # Save outputs
        self.logger.info('Saving centroids, assignments, figure')
        self._save_kclust(clusterer, 'kmeans', self.vids_dirpath, self.n, self.w, self.ds, self.max_nframes, self.pred_fn, k, it, r)

    def cluster_ts_kmedoids(self, data, dist_matrix, k, it):
        # Cluster
        self.logger.info('K-means clustering: k={}, it={}'.format(k, it))
        clusterer = ts_cluster(num_clust=k)
         # TODO: Debug / investigate this. See solution 1 in commit message about main problem with centroids
        self.logger.info('Trying to load DTW-based distance matrix')
        # clusterer.k_means_clust_modifcentroids(data, dist_matrix, it, 2, r)
        clusterer.k_medoids_clust(data, dist_matrix, it)

        # Un-normalize so that saved outputs -- centroids are of sentiment in range 0,1
        # Mean and std is per time series, i.e. one per video
        # Not sure if this is the 'right' way to do it, but just take the mean of the mean and std across all videos
        mean = self._load_ts_mean(self.vids_dirpath, self.n, self.w, self.ds, self.max_nframes, self.pred_fn)
        std = self._load_ts_std(self.vids_dirpath, self.n, self.w, self.ds, self.max_nframes, self.pred_fn)
        mean = mean.mean()
        std = std.mean()
        for i, c in enumerate(clusterer.centroids):
            clusterer.centroids[i] = list((np.array(c) * std) + mean)
        for i, m in enumerate(clusterer.medoids):
            clusterer.medoids[i] = list((np.array(m) * std) + mean)

        # Save outputs
        self.logger.info('Saving centroids, assignments, figure')
        self._save_kclust(clusterer, 'kmedoids', self.vids_dirpath, self.n, self.w, self.ds, self.max_nframes, self.pred_fn, k, it, None)

    def cluster_ts_hierarchical(self, data):
        """
        Parameters
        ----------
        data: np of dimension [num_timeseries, max_len]


        """
        clusterer = HierarchicalCluster()
        clusterer.cluster(data)

    def cluster_ts_hdbscan(self, data, r, mcs, ms):
        """
        Compute HDBSCAN cluster based on DTW distance matrix

        Parameters
        ----------
        data: np array of dimension (num_timeseries, max_leN)
        r: int, window size for LB_Keogh
        mcs: int, minimum_cluster_size
        ms: int, min_samples, # larger value is more conservative

        Saves
        -----
        Clusterer with fields: labels_, probabilities_, cluster_persistence_, condensed_tree_, single_linkage_tree_,
        minimum_spanning_tree_, outlier_scores_.
            - See: http://hdbscan.readthedocs.io/en/latest/api.html
        """
        self.logger.info('Clustering using HDBSCAN')

        # Load distance matrix if previously computed
        self.logger.info('Trying to load DTW-based distance matrix')
        dist_matrix = self._try_load_dtw_dist_matrix(self.vids_dirpath, self.n, self.w, self.ds, self.max_nframes, self.pred_fn, r)
        if dist_matrix is None:
            self.logger.info('No matrix found, creating DTW-based distance matrix - r:{}'.format(r))
            # Create DTW distance matrix sequentially. TODO: deprecate this
            # dist_matrix = np.zeros([len(data), len(data)])
            # for i in range(len(data)):
            #     self.logger.info(i)
            #     for j in range(i+1, len(data)):       # only calculate upper triangle
            #         # dist_matrix[i][j] = fastdtw_dist(data[i], data[j])
            #         dist_matrix[i][j] = LB_Keogh(data[i], data[j], r) if i != j else 0.0
            #         # dist_matrix[i][j] = DTWDistance(data[i], data[j], 5) if i != j else 0.0
            #         # print i, j, dist_matrix[i][j]

            # Create DTW distance matrix in parallel
            # data = data[0:10]
            dist_matrix = np.zeros([len(data), len(data)])

            def callback(i, j, future):
                dist_matrix[i][j] = future.result()
                # print 'Distance matrix entries not zero: {}'.format((dist_matrix != 0).sum())

            indices = [(i,j) for i in range(len(data)) for j in range(i+1, len(data))]      # only calculate upper triangle
            with ProcessPoolExecutor(max_workers=4) as executer:
                fs = []
                for i, j in indices:
                    future = executer.submit(LB_Keogh, data[i], data[j], r)
                    future.add_done_callback(partial(callback, i, j))
                    fs.append(future)
                wait(fs)

            # Save distance matrix
            self.logger.info('Saving DTW-based distance matrix')
            self._save_dtw_dist_matrix(dist_matrix, self.vids_dirpath, self.n, self.w, self.ds, self.max_nframes, self.pred_fn, r)

        # Cluster
        self.logger.info('Clustering')
        # mcs = mcs if mcs else len(data) / 20
        # ms = ms if ms else len(data) / 40
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs,
                                    min_samples=ms,
                                    metric='precomputed')  #,
                                    # gen_min_span_tree=True)
        clusterer.fit(dist_matrix)

        # Some logs
        self.logger.info('Number of clusters: {}'.format(clusterer.cluster_persistence_.shape[0]))
        self.logger.info('Cluster persistence: {}'.format(clusterer.cluster_persistence_))

        # Save
        self.logger.info('Saving clusterer')
        self._save_hdbscan(clusterer, self.vids_dirpath, self.n, self.w, self.ds, self.max_nframes, self.pred_fn, r, mcs, ms)

        # Robust single linkage hierarchical clustering
        # TODO: remove this. Not necessary? SingleLinkage used in HDBSCAN. If we want tree, use the single_linkage_tree_ field in clusterer
        # clusterer = hdbscan.RobustSingleLinkage(cut=0.125, k=7)
        # cluster_labels = clusterer.fit_predict(dist_matrix)
        # hierarchy = clusterer.cluster_hierarchy_
        # alt_labels = hierarchy.get_clusters(0.100, 5)
        # hierarchy.plot()
        # plt.savefig('tmp.png')

    ####################################################################################################################
    # Compute and save distances
    ####################################################################################################################
    def compute_kclust_error(self, alg, vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r):
        """
        - Compute sum_over_series{DTWDistance(series, centroid} for each k., basically WCSS
        - alg: kmeans or kmedoids
        - Used to determine optimal k
        - k is a comma-separated list
        """
        k2error = {}
        for cur_k in k.split(','):
            cur_k = int(cur_k)
            error = 0.0
            try:
                ts_dists = self._load_ts_dists(alg, vids_dirpath, n, w, ds, max_nframes, pred_fn, cur_k, it, r)
                for c_idx, c_members in ts_dists.items():
                    for m_idx, dist in c_members.items():
                        error += dist
                error /= float(n)
                k2error[cur_k] = error
                self.logger.info('k: {}, error: {}'.format(cur_k, error))
            except Exception as e:
                print e

        self._save_kclust_error(k2error, alg, vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r)

    ####################################################################################################################
    # Compute and save distances
    ####################################################################################################################
    # TODO: Deprecate this. ts_dists are now saved part of dtw_kmeans
    def compute_kclust_clusters_ts_dists(self, alg, vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r):
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

        for cur_k in k.split(','):
            print '=' * 100
            self.logger.info('k: {}'.format(cur_k))
            cur_k = int(cur_k)

            try:
                # Load data
                ts = self._load_ts(vids_dirpath, n, w, ds, max_nframes, pred_fn)
                centroids = self._load_centroids(vids_dirpath, n, w, ds, max_nframes, pred_fn, cur_k, it, r)
                assignments = self._load_assignments(vids_dirpath, n, w, ds, max_nframes, pred_fn, cur_k, it, r)

                # Compute
                ts_dists = {}
                for c_idx, members in assignments.items():
                    ts_dists[c_idx] = {}
                    i = 0
                    for m_idx in members:
                        # ts_dists[c_idx][m_idx] =
                        ts_dists[c_idx][m_idx] = LB_Keogh(centroids[c_idx], ts[m_idx], r)

                        if i  % 100 == 0:
                            self.logger.info('centroid: {} - {} distances computed'.format(c_idx, i))
                        i += 1

                # Save
                self._save_ts_dists(ts_dists, alg, vids_dirpath, n, w, ds, max_nframes, pred_fn, cur_k, it, r)

            except Exception as e:
                self.logger.info(e)

    ####################################################################################################################
    # Analyze coherence of 'groups' (combinations of different metadata, e.g. genre + year)
    ####################################################################################################################
    def analyze_group_coherence(self, vids_dirpath, n, w, ds, max_nframes, pred_fn, r):
        """
        Analyze coherence 'groups' (combinations of different metadata, e.g. genre + year)
        """
        MIN_GROUP_SIZE = 5

        self.logger.info('Analyzing group coherence')
        group2coherence = defaultdict(dict)

        # Load time series, title mappings
        ts = self._load_ts(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        ts_idx2title = self._load_ts_idx2title(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        title2ts_idx = {v:k for k,v in ts_idx2title.items()}

        # Get all groups, filter to those with at least MIN_GROUP_SIZE members
        group2titles, title2dirpath = self.get_groups(os.path.basename(vids_dirpath))
        self.logger.info('{} groups'.format(len(group2titles)))
        group2titles = {group: titles for group, titles in group2titles.items() if len(titles) > MIN_GROUP_SIZE}
        self.logger.info('{} groups have at least {} members'.format(len(group2titles), MIN_GROUP_SIZE))
        self.logger.info('Calculating coherence for {} groups'.format(len(group2titles)))

        for group, titles in group2titles.items():
            self.logger.info('Group: {}, num_videos: {}'.format(group, len(titles)))
            coherences = []         # parallel
            # coherence = 0.0       # sequential
            npairs = 0
            valid_titles = set()

            # Parallel
            def callback(future):
                coherences.append(future.result())

            indices = [(i,j) for i in range(len(titles)) for j in range(i+1, len(titles))]
            with ProcessPoolExecutor(max_workers=4) as executer:
                fs = []
                for i, j in indices:
                    title1 = titles[i]
                    title2 = titles[j]
                    if (title1 in title2ts_idx) and (title2 in title2ts_idx):
                        npairs += 1
                        valid_titles.add(title1)
                        valid_titles.add(title2)
                        future = executer.submit(LB_Keogh, ts[title2ts_idx[title1]], ts[title2ts_idx[title2]], r)
                        future.add_done_callback(callback)
                        fs.append(future)
                wait(fs)
            coherence = sum(coherences) / float(npairs)

            # # Sequential
            # for i in range(len(titles)):
            #     for j in range(i+1, len(titles)):
            #         title1 = titles[i]
            #         title2 = titles[j]
            #         if (title1 in title2ts_idx) and (title2 in title2ts_idx):
            #             valid_titles.add(title1)
            #             valid_titles.add(title2)
            #             coherence += LB_Keogh(ts[title2ts_idx[title1]], ts[title2ts_idx[title2]], r)
            #             npairs += 1
            # coherence /= float(npairs)

            # Add to data structure
            group2coherence[group]['coherence'] = coherence
            group2coherence[group]['titles'] = valid_titles
            self.logger.info('.............coherence: {}'.format(coherence))
            self.logger.info('.............valid titles: {}'.format(len(valid_titles)))

        # Save group2coherence
        self.logger.info('Saving group coherence')
        self._save_group_coherence(group2coherence, vids_dirpath, n, w, ds, max_nframes, pred_fn, r)

    def get_groups(self, fmt):
        if fmt == 'films':
            return self.get_films_groups()
        elif fmt == 'shorts':
            return self.get_shorts_groups()

    def get_films_groups(self):
        """
        Return
            group2titles: dict, key is tuple of group tags (genre, year), value is list of titles
            title2dirpath: dict, key is title, value is path to directory
        """
        self.logger.info('Getting films groups')
        conn = sqlite3.connect(VIDEOPATH_DB)
        with conn:
            cur = conn.cursor()
            rows = cur.execute("SELECT title, dirpath FROM VideoPath WHERE category=='films'")
            title2dirpath = {row[0]: row[1] for row in rows}

        video2metadata = pickle.load(open(VIDEOMETADATA_DB, 'rb'))
        group2titles = defaultdict(list)
        for title, dirpath in title2dirpath.items():
            # print title, dirpath
            if title in video2metadata:
                # print 'ok'
                genres = video2metadata[title]['genres']
                for g in genres:
                    group2titles[(g)].append(title)

        # print video2metadata.keys()

        return group2titles, title2dirpath

    def get_shorts_groups(self):
        pass

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
                    and (len(os.listdir(os.path.join(root, 'frames'))) > 0) \
                    and (PRED_FN in os.listdir(os.path.join(root, 'preds'))):
                vidpaths.append(root)

        return vidpaths

    def _save_ts(self, ts, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        params_str = self._get_TS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        path = self._get_ts_path(params_str)
        with open(path, 'wb') as f:
            pickle.dump(ts, f, protocol=2)

    def _save_ts_mean(self, mean, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        params_str = self._get_TS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        path = self._get_ts_mean_path(params_str)
        with open(path, 'wb') as f:
            pickle.dump(mean, f, protocol=2)

    def _save_ts_std(self, std, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        params_str = self._get_TS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        path = self._get_ts_std_path(params_str)
        with open(path, 'wb') as f:
            pickle.dump(std, f, protocol=2)

    def _save_ts_idx2title(self, ts_idx2title, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        params_str = self._get_TS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        path = self._get_ts_idx2title_path(params_str)
        with open(path, 'w') as f:
            pickle.dump(ts_idx2title, f, protocol=2)

    def _load_ts(self, vids_dirpath, n, w, ds, max_nframes, pred_fn):
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

    def _load_ts_idx2title(self, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        params_str = self._get_TS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        path = self._get_ts_idx2title_path(params_str)
        ts_idx2title = pickle.load(open(path, 'rb'))
        return ts_idx2title

    def _load_ts_mean(self, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        params_str = self._get_TS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        path = self._get_ts_mean_path(params_str)
        mean = pickle.load(open(path, 'rb'))
        return mean

    def _load_ts_std(self, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        params_str = self._get_TS_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn)
        path = self._get_ts_std_path(params_str)
        std = pickle.load(open(path, 'rb'))
        return std

    def _save_kclust(self, clusterer, alg, vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r):
        """
        Save centroids, assignments, plots, and ts-distsfor kmeans
        """
        params_str = self._get_KCLUST_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r)
        centroids_path = self._get_centroids_path(alg, params_str)
        assignments_path = self._get_assignments_path(alg, params_str)
        with open(centroids_path, 'wb') as f:
            pickle.dump(clusterer.centroids, f, protocol=2)
        with open(assignments_path, 'wb') as f:
            pickle.dump(clusterer.assignments, f, protocol=2)

        # Plot centroids
        for i, c in enumerate(clusterer.centroids):
            plt.plot(c, label=i)
        plt.legend()
        plt.savefig(os.path.join(OUTPUTS_PATH, 'imgs', '{}-centroids_{}.png'.format(alg, params_str)))
        plt.gcf().clear()               # clear figure so for next k

        # Plot medoids if kmedoids
        if alg == 'kmedoids':
            for i, m in enumerate(clusterer.medoids):
                plt.plot(m, label=i)
            plt.legend()
            plt.savefig(os.path.join(OUTPUTS_PATH, 'imgs', '{}-medoids_{}.png'.format(alg, params_str)))
            plt.gcf().clear()

        # Ts dists to centroids
        ts_dists_path = self._get_ts_dists_path(alg, params_str)
        with open(ts_dists_path, 'wb') as f:
            pickle.dump(clusterer.ts_dists, f, protocol=2)

        # Some extra logging
        for centroid_idx, assignments in clusterer.assignments.items():
            self.logger.info('Centroid {}: {} series'.format(centroid_idx, len(assignments)))

    def _save_ts_dists(self, ts_dists, alg, vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r):
        params_str = self._get_KCLUST_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r)
        path = self._get_ts_dists_path(alg, params_str)
        with open(path, 'wb') as f:
            pickle.dump(ts_dists, f, protocol=2)

    def _save_kclust_error(self, k2error, alg, vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r):
        params_str = self._get_KCLUST_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r)
        path = self._get_kclust_error_path(alg, params_str)
        with open(path, 'wb') as f:
            pickle.dump(k2error, f, protocol=2)

        # Plot
        plt.plot(k2error.keys(), k2error.values())
        plt.savefig(os.path.join(OUTPUTS_PATH, 'imgs', '{}-error_{}.png'.format(alg, params_str)))
        plt.gcf().clear()


    def _save_hdbscan(self, clusterer, vids_dirpath, n, w, ds, max_nframes, pred_fn, r, mcs, ms):
        """
        Save clusterer for hdbscan
        """
        params_str = self._get_HDBSCAN_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn, r, mcs, ms)
        path = self._get_hdbscan_path(params_str)
        with open(path, 'wb') as f:
            pickle.dump(clusterer, f, protocol=2)

        # Plots
        # # print clusterer.condensed_tree_.to_pandas().head()
        # clusterer.single_linkage_tree_.plot()
        # plt.savefig('outputs/cluster/imgs/sltree.png')
        # plt.gcf().clear()
        # clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette('deep', 8))

    def _save_dtw_dist_matrix(self, dtw_dist_matrix, vids_dirpath, n, w, ds, max_nframes, pred_fn, r):
        params_str = self._get_DIST_MATRIX_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn, r)
        path = self._get_dtw_dist_matrix_path(params_str)
        with open(path, 'wb') as f:
            pickle.dump(dtw_dist_matrix, f, protocol=2)

    def _try_load_dtw_dist_matrix(self, vids_dirpath, n, w, ds, max_nframes, pred_fn, r):
        params_str = self._get_DIST_MATRIX_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn, r)
        path = self._get_dtw_dist_matrix_path(params_str)
        if os.path.exists(path):
            dist_matrix = pickle.load(open(path, 'rb'))
            return dist_matrix
        else:
            return None

    def _load_centroids(self, alg, vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r):
        """
        Load centroids from k means clustering
        """
        clusters_str = self._get_KCLUST_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r)
        centroids_path = self._get_centroids_path(alg, clusters_str)
        centroids = pickle.load(open(centroids_path, 'rb'))
        return centroids

    def _load_assignments(self, alg, vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r):
        """
        Load assignments from k means clustering
        """
        clusters_str = self._get_KCLUST_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r)
        assignments_path = self._get_assignments_path(alg, clusters_str)
        assignments = pickle.load(open(assignments_path, 'rb'))
        return assignments

    def _load_ts_dists(self, alg, vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r):
        params_str = self._get_KCLUST_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r)
        path = self._get_ts_dists_path(alg, params_str)
        ts_dists = pickle.load(open(path, 'rb'))
        return ts_dists

    def _save_group_coherence(self, group2coherence, vids_dirpath, n, w, ds, max_nframes, pred_fn, r):
        params_str = self._get_GROUP_COHERENCE_STR_formatted(vids_dirpath, n, w, ds, max_nframes, pred_fn, r)
        path = self._get_group_coherence_path(params_str)
        with open(path, 'wb') as f:
            pickle.dump(group2coherence, f, protocol=2)

    def _get_TS_STR_formatted(self, vids_dirpath, n, w, ds, max_nframes, pred_fn):
        str = TS_STR.format(
            os.path.basename(vids_dirpath), n,
            w, ds, max_nframes,
            pred_fn[:-4])      # remove .csv
        return str

    def _get_KCLUST_STR_formatted(self, vids_dirpath, n, w, ds, max_nframes, pred_fn, k, it, r):
        str = KCLUST_STR.format(
            os.path.basename(vids_dirpath), n, w,
            ds, max_nframes, pred_fn[:-4],
            k, it, r)      # remove .csv
        return str

    def _get_DIST_MATRIX_STR_formatted(self, vids_dirpath, n, w, ds, max_nframes, pred_fn, r):
        str = DIST_MATRIX_STR.format(
            os.path.basename(vids_dirpath), n, w,
            ds, max_nframes, pred_fn[:-4], r)
        return str

    def _get_GROUP_COHERENCE_STR_formatted(self, vids_dirpath, n, w, ds, max_nframes, pred_fn, r):
        str = GROUP_COHERENCE_STR.format(
            os.path.basename(vids_dirpath), n, w,
            ds, max_nframes, pred_fn[:-4], r)
        return str

    def _get_HDBSCAN_STR_formatted(self, vids_dirpath, n, w, ds, max_nframes, pred_fn, r, mcs, ms):
        str = HDBSCAN_STR.format(
            os.path.basename(vids_dirpath), n, w,
            ds, max_nframes, pred_fn[:-4],
            r, mcs, ms)      # remove .csv
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

    def _get_centroids_path(self, alg, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', '{}-centroids_{}.pkl'.format(alg, params_str))
        return path

    def _get_assignments_path(self, alg, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', '{}-assignments_{}.pkl'.format(alg, params_str))
        return path

    def _get_ts_dists_path(self, alg, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', '{}-ts-dists_{}.pkl'.format(alg, params_str))
        return path

    def _get_kclust_error_path(self, alg, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', '{}-error_{}.pkl'.format(alg, params_str))
        return path

    def _get_dtw_dist_matrix_path(self, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', 'dtw-dist-matrix_{}.pkl'.format(params_str))
        return path

    def _get_group_coherence_path(self, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', 'group-coherence_{}.pkl'.format(params_str))
        return path

    def _get_hdbscan_path(self, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', 'hdbscan_{}.pkl'.format(params_str))
        return path

    def _get_logger(self):
        """Return logger, where path is dependent on mode (train/test), arch, and obj"""
        logs_path = os.path.join(os.path.dirname(__file__), 'logs')
        _, logger = setup_logging(save_path=os.path.join(logs_path, 'analysis.log'))
        return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster time series')

    # Action to take
    parser.add_argument('--prepare_ts', dest='prepare_ts', action='store_true', default=False)
    parser.add_argument('--cluster_ts', dest='cluster_ts', action='store_true', default=False)
    parser.add_argument('-m', '--method', dest='method', default=None, help='kmeans,kmedoids,hierarchical,hdbscan')
    parser.add_argument('--compute_kclust_error', dest='compute_kclust_error', action='store_true', default=False)
    parser.add_argument('--compute_kclust_clusters_ts_dists', dest='compute_kclust_clusters_ts_dists', action='store_true', default=False)
    parser.add_argument('--analyze_group_coherence', dest='analyze_group_coherence', action='store_true', default=False)

    # Time serise data parameters
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
    parser.add_argument('-r', dest='r', type=int, default=None, help='LB_Keogh window size')
    parser.add_argument('-k', dest='k', default=None, help='k-means: list of comma-separated k to evaluate')
    parser.add_argument('-it', dest='it', type=int, default=None, help='k-means: number of iterations')
    parser.add_argument('-mcs', dest='mcs', type=int, default=None,
                        help='HDBSCAN: min_cluster_size. If none, use....')
    parser.add_argument('-ms', dest='ms', type=int, default=None,
                        help='HDBSCAN: min_samples (larger is more conservative clustering). If None, use....')


    cmdline = parser.parse_args()

    analysis = Analysis()
    if cmdline.prepare_ts:
        analysis.prepare_ts(cmdline.vids_dirpath, cmdline.w,
                                     cmdline.ds, cmdline.max_nframes, cmdline.pred_fn)
    elif cmdline.cluster_ts:
        ts = analysis._load_ts(cmdline.vids_dirpath, cmdline.n, cmdline.w,
                              cmdline.ds, cmdline.max_nframes, cmdline.pred_fn)
        analysis.cluster_ts(ts, cmdline.method, cmdline.r, cmdline.k, cmdline.it, cmdline.mcs, cmdline.ms)
    elif cmdline.compute_kclust_error:
        analysis.compute_kclust_error(cmdline.method, cmdline.vids_dirpath, cmdline.n, cmdline.w,
                                      cmdline.ds, cmdline.max_nframes, cmdline.pred_fn,
                                      cmdline.k, cmdline.it, cmdline.r)
    elif cmdline.compute_kclust_clusters_ts_dists:
        analysis.compute_kclust_clusters_ts_dists(cmdline.method, cmdline.vids_dirpath, cmdline.n,
                                                           cmdline.w, cmdline.ds,
                                                           cmdline.max_nframes, cmdline.pred_fn,
                                                           cmdline.k, cmdline.it, cmdline.r)
    elif cmdline.analyze_group_coherence:
        analysis.analyze_group_coherence(cmdline.vids_dirpath, cmdline.n,
                                         cmdline.w, cmdline.ds, cmdline.max_nframes, cmdline.pred_fn, cmdline.r)