# Prepare the data (parse, convert to tfrecords, download, etc.)

import argparse
from collections import defaultdict, Counter
import datetime
from fuzzywuzzy import fuzz
import io
import json
import os
import pandas as pd
import pickle
from pprint import pprint
import re
import shutil
import sqlite3
import subprocess
import time

from core.predictions.utils import detect_peaks, smooth
from core.utils.CreditsLocator import CreditsLocator
from core.utils.MovieReader import MovieReader
from core.utils.utils import VID_EXTS, VIZ_SENT_PRED_FN, AUDIO_SENT_PRED_FN

# Videos path
VIDEOS_PATH = 'data/videos'
HIGHLIGHTS_PATH = 'data/videos/highlights'

# CMU Movie Summary path
CMU_PATH = 'data/CMU_movie_summary/MovieSummaries/'

# Video DBs
VIDEOPATH_DB = 'data/db/VideoPath.db'
VIDEOMETADATA_DB = 'data/db/VideoMetadata.pkl'


########################################################################################################################
########################################################################################################################
########################################################################################################################
#
# Videos
#
########################################################################################################################
########################################################################################################################
########################################################################################################################

########################################################################################################################
# Frames
########################################################################################################################
def save_video_frames(vids_dir, sr):
    """
    Loop over subdirs within vids_dir and save frames to subdir/frames/

    Parameters
    ----------
    vids_dir: directory within VIDEOS_PATH that contains sub-directories, each which may contain a movie
        e.g. films/animated/
        TODO: refactor this to os.walk, so you can just pass in data/videos/...
    sr: float - sample rate (e.g. 1 means take a frame at every second)
    """

    # vid_exts = ['mp4', 'avi']
    mr = MovieReader()

    vids_path = os.path.join(VIDEOS_PATH, vids_dir)
    print vids_path
    ext2count = defaultdict(int)
    i = 0
    successes = []
    for vid_name in [d for d in os.listdir(vids_path) if not d.startswith('.')]:
        # Skip if frames/ already exists and has some frames in it
        if os.path.exists(os.path.join(vids_path, vid_name, 'frames')):
            if len(os.listdir(os.path.join(vids_path, vid_name, 'frames'))) != 0:
                continue

        # Get the actual video file, while also removing any sample video files if they are there
        vid_dirpath = os.path.join(vids_path, vid_name)
        files = os.listdir(vid_dirpath)
        movie_file = None
        vid_ext = None
        for f in files:
            if 'sample' in f.lower():
                try:
                    os.remove(os.path.join(vid_dirpath, f))
                except Exception as e:
                    print 'Removed sample file {} for {}'.format(f, vid_name)
        for f in files:
            for ext in VID_EXTS:
                if f.endswith(ext):
                    movie_file = f
                    ext2count[ext] += 1
                    vid_ext = ext

        # Try to save frames for video file
        if movie_file:
            print '=' * 100
            print 'Video: {}'.format(vid_name)
            print 'Format: {}'.format(vid_ext)
            movie_path = os.path.join(vid_dirpath, movie_file)
            try:
                mr.write_frames(movie_path, sample_rate_in_sec=sr)
                i += 1
                successes.append(vid_name)
                # TODO: should check number of frames -- sometimes only a few saved, there's an error going through file
            except Exception as e:
                print e

    print '=' * 100
    print 'Created frames for {}'.format(successes)
    print 'Extension counts: {}'.format(ext2count)      # will only be for movies without frames/
    print 'Created frames for {} videos'.format(i)

def convert_avis_to_mp4s(vids_dir):
    """
    Convert videos from avi to mp4

    Notes
    -----
    Frames cannot be extracted from avi files using MovieReader currently, thus convert to more standard mp4 format

    Parameters
    ----------
    vids_dir: directory within VIDEOS_PATH that contains sub-directories, each which may contain a movie
    """
    vids_path = os.path.join(VIDEOS_PATH, vids_dir)
    vid_dirs = [d for d in os.listdir(vids_path) if not d.startswith('.')]
    for vid_dir in vid_dirs:
        # Find avi file if it exists
        vid_dirpath = os.path.join(vids_path, vid_dir)
        filenames = os.listdir(vid_dirpath)
        avi_fn = None
        for fn in filenames:
            if fn.endswith('avi'):
                avi_fn = fn

        # Convert to mp4, clean up if it succeeds
        if avi_fn:
            try:
                print '=' * 100
                print '=' * 100
                print '=' * 100
                print 'Found avi file to convert for: {}'.format(vid_dir)
                mp4_fn = avi_fn.split('.avi')[0] + '.mp4'
                avi_path = os.path.join(vid_dirpath, avi_fn)
                mp4_path = os.path.join(vid_dirpath, mp4_fn)
                bash_command = ['avconv', '-i'] + [avi_path] + ['-c:v', 'libx264', '-c:a', 'copy'] + [mp4_path]
                # Not using split() on bash command string because path's may have spaces
                print bash_command
                subprocess.call(bash_command, stdout=subprocess.PIPE)
                print 'Done converting, will remove avi file'
                os.remove(avi_path)
            except Exception as e:
                print e

def save_credits_index(vids_dir, overwrite_files=False):
    """
    Save index of frames/ for when credits start
    """
    cl = CreditsLocator(overwrite_files=overwrite_files)
    vids_path = os.path.join(VIDEOS_PATH, vids_dir)
    vid_dirs = [d for d in os.listdir(vids_path) if not d.startswith('.')]
    not_located = []
    for vid_dir in vid_dirs:
        print '=' * 100
        print vid_dir
        located = cl.locate_credits(os.path.join(vids_path, vid_dir))
        if not located:
            not_located.append(vid_dir)

    print '=' * 100
    print 'Credits not located for {} movies:'.format(len(not_located))
    pprint(sorted(not_located))

########################################################################################################################
# VideoPath DB
########################################################################################################################
def extract_highlight_clips(vids_dirpath, overwrite, verbose):
    """
    Use the saved visual and audio predictions to extract and save clips from peaks and valleys. These will be labeled
    for ground truth. Currently uses the visual sentiment and negative predictions.

    Parameters
    ----------
    vids_dirpath: str, e.g. 'data/videos/films'
    overwrite: boolean - overwrite clips. Required in order to get ffmpeg to run without having to press 'y'.
    verbose: boolean - pipe ffmpeg process output to stdout

    Notes
    -----
    - Currently skips non-mp4 videos. Don't need clips on all movies anyway.
    - Parameters
        - MPD: the mpd should be a reasonably high value, say 600 (seconds) to a) prevent extracting scenes that are
        too close to each other, and b) prevent extracting too many scenes.
        - MERGE_SR: if > 1 (say 2 or 3) MERGE_MIN_DIST should probably be 0
        - MPH: preds need to be range_norm()'d
    """
    CLIP_START_NPREDS = 300     # number of predictions at start to ignore
    CLIP_END_NPREDS = 600       # number of predictions at end to ignore
    WINDOW_LEN = 600            # for smoothing
    SHOW_EXTREMA = False         # just used during debugging detect_peaks
    MPD = 600                   # min peak distance in detect_peaks
    MPH = 0.85                  # Get peaks at least as large as this value, valleys as low as 1-MPH
    MERGE_MIN_DIST = 0          # minimum distance between extrema when merging
    MERGE_SR = 1                # take every _ point from merged extrema in order to reduce points.
    CLIP_LENGTH = 30
    OUT_WIDTH = 640

    def find_movie_file(files):
        """
        Find movie file (name) in list of files if it exists
        """
        for f in files:
            for ext in VID_EXTS:
                if f.endswith(ext):
                    return f
        return None

    def get_timestamp_str(seconds):
        """Return '0h3m01s' for 181"""
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        timestamp_str = '{}h{:02}m{:02}s'.format(int(h),int(m),int(s))
        return timestamp_str

    def range_norm(preds):
        """
        Given np array of preds, normalize each value to [0,1] by subtracting min and dividing by range
        """
        return (preds - preds.min()) / (preds.max() - preds.min())

    def merge_extrema(peaks_valleys, modalities, extrema_types, min_dist=MERGE_MIN_DIST, sr=MERGE_SR):
        """
        Combine and dedupe lists of extremas

        Parameters
        ----------
        peaks_valleys: list of list of indices - each sublist stores index of an extrema
        modalities: list of strs - modality  of each sublist in peaks_and_valleys, e.g. audio
        extrema_types: list of strs - extrema_type of each sublist in peaks_and_valleys, e.g. valley
        min_dist: int - minimum distance between end of clip1 and start of clip2
        sr: int - take every other, every third, every sr point
            - another quick way to reduce number of points (min_dist also reduces, but has its own problems --
              see todo in below comments)

        Returns
        -------
        extrema: list of indices where extrema occur
        tags: list of lists. tag[i] stores names for extrema i
            - e.g. some extrema might be both audio-peak and visual-peak

        Methodology
        -----------
        Dealing with overlaps
            1) Same modality, same extremum-type (e.g. audio-valley audio-valley):
            - The minimum-peak-distance (mpd flag in detect_peaks) should be greater than the clip length, which means
            this shouldn't happen. In fact, the mpd should be a reasonably high value, say 600 (seconds)
            to a) prevent extracting scenes that are too close to each other, and b) prevent extracting
            too many scenes.

            2) Same modality, different extremum-type (e.g. audio-peak audio-valley):
            - Throw out these extrema.
            - Reasoning: a) scene might be ambiguous, b) not particularly interested in 'diagnosing' these because
              next point is main example of when this happens (i.e. I've already diagnosed it and it's not really an
              issue) per se.
            - Case: after looking at some plots with SHOW_EXTREMA=True, this happens sometimes when there is a minor
            peak followed by a very minor valley (minor because the smoothing would prevent drastic changes, i.e.
            no large peaks followed by large valleys).

            3) Different modality (e.g. audio-peak visual-valley; audio-valley visual-valley)
            - After rules 1) and 2), this is the only type of overlap that can happen.
            - Note that it could still be the case that A overlaps with B, which overlaps with C, etc.
                - Example of such a 'connected-component': A = audio-valley, B = visual-peak, C = audio-valley

            - In general, we want overlaps between modalities -- these are potentially more interesting because
            either the two modalities match (both peaks or both valleys), or they differ. In the former case, we would
            like to confirm the extrema. In the later, we would like to diagnose the clip.

            - When the connected-component size == 2: take the index as the average of indices of the overlapping clips
            - When the connected-copmonent size > 2: ignore these extrema
                - Reasoning:
                    a) scene might be ambiguous (e.g. audio-valley, visual-peak, audio-peak)
                    b) taking the index as some average of 3+ may miss out on context and what made that point an
                       extrema.
                        - Example: [30,59,88]. Ranges are [15-45], [44-74], [73-105]. No way to keep same clip length
                        and include at least 50% of each range

        Enforce minimum distance between clips, e.g. min_dist = 30 --> [0,30], [61, 91]
            - Step 1: changing the overlap boolean inequaity to <= min_dist + CLIP_LENGTH
            - Step 2: Think this would mean no overlaps though?
                - Can't take average of two clips that don't even actually overlap
                - So if overlap, just take the first one.
                    - TODO: However, this doesn't take into account the magnitude of each extrema. Perhaps
                    the second one should be chosen if it is a 'better' peak.
                    - TODO: Each peak should have a peak score, which is a function of the preds curve and the modality.
            - Probably not that many overlaps anyway though to be honest. Given that I'm trying to find major
            peaks and valleys using detect_peaks, and 30 seconds is an awfully small window for the visual
            and audio to overlap, even if the models were dead accurate.


        Test
        ----
            CLIP_LENGTH = 5

            def test(preds, modalities, extrema_types,):
              extrema, tags = merge_extrema(preds, modalities, extrema_types)
              print '=' * 50
              print extrema
              print tags

            # Test - no overlap, super simple
            modalities = ['visual', 'audio']
            extrema_types = ['peak', 'valley']
            A = [0]
            B = [10]
            test([A, B], modalities, extrema_types)

            # Test - no overlap, longer
            modalities = ['visual', 'audio']
            extrema_types = ['peak', 'valley']
            A = [0, 20, 40, 60, 80]
            B = [10, 30, 50, 70, 90]
            test([A, B], modalities, extrema_types)

            # Test - same modality overlap
            modalities = ['visual', 'visual']
            extrema_types = ['peak', 'valley']
            A = [0]
            B = [3]
            test([A, B], modalities, extrema_types)

            # Test - different modality overlap
            modalities = ['visual', 'audio']
            extrema_types = ['peak', 'valley']
            A = [0]
            B = [3]
            test([A, B], modalities, extrema_types)

            # Test - 3 overlap
            modalities = ['visual', 'audio']
            extrema_types = ['peak', 'valley']
            A = [0, 8]
            B = [4]
            test([A, B], modalities, extrema_types)

            # Test - overlaps
            modalities = ['visual', 'visual', 'audio', 'audio']
            extrema_types = ['peak', 'valley', 'peak', 'valley']
            A = [0, 20]
            B = [2, 50, 100]
            C = [22, 30, 52]
            D = [54, 102, 150]
            test([A, B, C, D], modalities, extrema_types)

        Test outputs
        ------------
            ==================================================
            [0.0, 10.0]
            [['visual-peak'], ['audio-valley']]
            ==================================================
            [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
            [['visual-peak'], ['audio-valley'], ['visual-peak'], ['audio-valley'], ['visual-peak'], ['audio-valley'],
            ['visual-peak'], ['audio-valley'], ['visual-peak'], ['audio-valley']]
            ==================================================
            []
            []
            ==================================================
            [1.5]
            [['visual-peak', 'audio-valley']]
            ==================================================
            []
            []
            ==================================================
            [21.0, 30.0, 101.0, 150.0]
            [['visual-peak', 'audio-peak'], ['audio-peak'], ['visual-valley', 'audio-valley'], ['audio-valley']]
        """
        # Combine extrema points in sorted order
        for i, lst in enumerate(peaks_valleys):            # tag each item in each sublist with its name
            peaks_valleys[i] = [(idx, modalities[i], extrema_types[i]) for idx in lst]
        peaks_valleys = [item for sublist in peaks_valleys for item in sublist]     # flatten
        peaks_valleys = sorted(peaks_valleys)

        # Merge, dedup, remove close extrema, etc.
        ccs = []        # list of (start, tags), where tags is a list
        last_extrema_start = float('-inf')
        last_extrema_modality = None
        cur_cc = []     # list of (start, tag)
        for i in range(len(peaks_valleys)):
            start, modality, extremum_type = peaks_valleys[i]
            overlap = (start - last_extrema_start) <= CLIP_LENGTH + min_dist
            if overlap:
                if min_dist > 0:                            # clips don't actually 'overlap', just ignore the second
                    continue
                else:
                    if last_extrema_modality == modality:     # remove extrema (i.e. remove cur_cc)
                        cur_cc = []
                        last_extrema_start = ccs[-1][0] if len(ccs) > 0 else float('-inf')
                    else:       # different modality
                        if len(cur_cc) == 1:
                            cur_cc.append((start, modality + '-' + extremum_type))
                            last_extrema_start = start
                            last_extrema_modality = modality
                        else:   # adding another one will result in cc_size of 3, so remove extrema (i.e. remove cur_cc)
                            cur_cc = []
                            last_extrema_start = ccs[-1][0] if len(ccs) > 0 else float('-inf')
            else:
                # No overlap, so add the 'current' (previous now) cc and update the current cc
                if len(cur_cc) > 0:         # will be 0 the first start pass through
                    cc_start = sum([s for s, t in cur_cc]) / float(len(cur_cc))
                    cc_tags = [t for s, t in cur_cc]
                    ccs.append((cc_start, cc_tags))

                cur_cc = [(start, modality + '-' + extremum_type)]       # extremum, tag
                last_extrema_start = start
                last_extrema_modality = modality

        # Finish up
        if len(cur_cc) > 0:
          cc_start = sum([start for start, tag in cur_cc]) / float(len(cur_cc))
          cc_tags = [tag for start, tag in cur_cc]
          ccs.append((cc_start, cc_tags))

        # Downsample
        ccs = ccs[0:len(ccs):sr]

        # Unzip ccs
        extrema = []
        tags = []
        for e, t in ccs:
          extrema.append(e)
          tags.append(t)

        return extrema, tags

    def filter_ends(preds, start_npreds=CLIP_START_NPREDS, end_npreds=CLIP_END_NPREDS):
        """
        Remove start and end of preds to ignore opening sequence of studio and end credits. Could refactor this to use
        the saved credits_idx, but I don't think this has to be very accurate. This function can help if we
        normalize predictions to be in [0,1] so that the beginning and end dips don't affect the range as much.
        Removes start_npreds predictions from start, end_npreds predictions from end.
        """
        return preds[start_npreds:len(preds)-end_npreds]

    def save_extrema_clips(extrema, tags, movie_path, movie, ext, verbose):
        """
        Extract clips from videos and save to folder

        Parameters
        -------
        extrema: list of indices where extrema occur
        tags: list of lists. tag[i] stores names for extrema i
            - e.g. some extrema might be both audio-peak and visual-peak
        movie_path: str -  path to movie
        movie: str - title of video
            - same title as in MoviePathDB, e.g. Frozen (2013)
        ext: str - video extension, e.g. mp4
        """

        for i, extremum in enumerate(extrema):
            # Get start and end time to extract
            # Take int(extremum) because it could be a float if there was an overlap
            start_time = CLIP_START_NPREDS + int(extremum) - (CLIP_LENGTH / 2)      # in seconds
            end_time = CLIP_START_NPREDS + int(extremum) + (CLIP_LENGTH / 2)

            # Get filename and path to save to
            # Include i so that we can natsort and get clips in chronological order easily
            cur_tags = '+'.join(tags[i])
            start_str = get_timestamp_str(start_time)
            end_str = get_timestamp_str(end_time)
            out_fn = '{}_{}_{}_s{}_e{}_l{}{}'.format(movie, i, cur_tags, start_str, end_str, CLIP_LENGTH, ext)
            out_path = os.path.join(out_dirpath, out_fn)

            # Create and call command
            cmd = ['ffmpeg', '-ss', str(start_time), '-t', str(CLIP_LENGTH), '-i', movie_path, '-vf', 'scale=640:-2',
                   '-crf', '29', '-async', '1', out_path]
            if overwrite:
                cmd.insert(1, '-y')
            if verbose:
                subprocess.call(cmd, stdout=subprocess.PIPE)
            else:
                FNULL = open(os.devnull, 'wb')
                subprocess.call(cmd, stdout=FNULL, stderr=subprocess.STDOUT)

    # Main function starts here -- find all videos with predictions and movies
    nvids = 0
    start_run_time = time.time()
    for root, dirs, files in os.walk(vids_dirpath):
        if 'preds' in dirs:
            viz_path = os.path.join(root, 'preds', VIZ_SENT_PRED_FN)
            audio_path = os.path.join(root, 'preds', AUDIO_SENT_PRED_FN)
            movie_file = find_movie_file(files)
            movie = os.path.basename(root.rstrip('/'))
            if os.path.exists(viz_path) and os.path.exists(audio_path) and movie_file:
                vid_start_run_time = time.time()
                if verbose:
                    print '=' * 100
                    print '=' * 100
                    print '=' * 100
                else:
                    print '=' * 100
                print 'Found data for {}'.format(root)

                # Get some info
                fn, ext = os.path.splitext(movie_file)
                movie_path = os.path.join(root, movie_file)
                out_dirpath = os.path.join(HIGHLIGHTS_PATH, movie)

                # Skip if not overwriting and highlights already exists
                if os.path.exists(out_dirpath):
                    if not overwrite:
                        print 'Skipping -- overwrite=false, highlights directory for movie already exists'
                        continue

                # Skip if extension isn't mp4
                if ext not in ['.mp4', '.MP4']:
                    print 'Skipping -- extension is {}, not mp4'.format(ext)
                    continue

                # Skip videos that aren't as wide as desired width (mostly avi's probably)
                cmd = 'ffprobe -v error -of flat=s=_ -select_streams v:0 -show_entries stream=width,height'.split(' ') + [movie_path]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                out, err = proc.communicate()           # 'streams_stream_0_width=1280\nstreams_stream_0_height=568\n'
                m = re.match(r'.+width=([0-9]+)\n.+', out)
                if (m is None) or (int(m.group(1)) < OUT_WIDTH):
                    print 'Skipping - video width is {}, less than {}'.format(m.group(1), OUT_WIDTH)
                    continue

                # Make directory to store highlight clips (if it doesn't already exist)
                if os.path.exists(out_dirpath):
                    # If it reaches this point, overwrite should always be true
                    # Leaving this if here for clarity
                    # The skipping if not overwrite is placed further up in the code to short-circuit earlier,
                    # and not create a directory if it's skipped because of extension of width reasons
                    if overwrite:
                        shutil.rmtree(out_dirpath)
                        os.mkdir(out_dirpath)
                else:
                    os.mkdir(out_dirpath)
                print 'Saving clips to {}'.format(out_dirpath)

                # Get peaks
                # TODO: skip if number of extrema too low or too high?
                # Minor to-do: column names / number of columns should be put in core/utils/utils.py GLOBALS?
                viz_preds = range_norm(filter_ends(smooth(pd.read_csv(viz_path).pos.values, window_len=WINDOW_LEN)))
                viz_peaks = detect_peaks(viz_preds, mpd=MPD, mph=MPH, edge=None, show=SHOW_EXTREMA)
                viz_valleys = detect_peaks(viz_preds, mpd=MPD, mph=MPH-1.0, edge=None, valley=True, show=SHOW_EXTREMA)
                audio_preds = range_norm(filter_ends(smooth(pd.read_csv(audio_path).Valence.values, window_len=WINDOW_LEN)))
                audio_peaks = detect_peaks(audio_preds, mpd=MPD, mph=MPH, edge=None, show=SHOW_EXTREMA)
                audio_valleys = detect_peaks(audio_preds, mpd=MPD, mph=MPH-1.0, edge=None, valley=True, show=SHOW_EXTREMA)

                # Merge, dedup, tag, etc. extrema
                extrema, tags = merge_extrema([viz_peaks, viz_valleys, audio_peaks, audio_valleys],
                                              ['visual', 'visual', 'audio', 'audio'],
                                              ['peak', 'valley', 'peak', 'valley'])

                # Save each highlight clip
                save_extrema_clips(extrema, tags, movie_path, movie, ext, verbose)

                nvids += 1

                # Stats
                print 'Done extracting {} clips:'.format(len(extrema))
                print 'Time elapsed for video: {:.2f} seconds'.format(time.time() - vid_start_run_time)
                print 'Extracted clips from {} videos'.format(nvids)


    print 'Total run time: {}'.format(time.time() - start_run_time)

########################################################################################################################
# VideoPath DB
########################################################################################################################
def create_videopath_db():
    """
    Create sqllite db storing information about video paths, formats, frames, etc.
    """
    def get_movie_fn_if_exists(files):
        for f in files:
            if 'sample' in f.lower():
                continue
            for ext in VID_EXTS:
                if f.endswith(ext):
                    return f
        return None

    def get_dataset_name_from_dir(dir):
        # MovieQA_full_movies -> MovieQA
        if dir == 'MovieQA_full_movies':
            return 'MovieQA'
        elif dir == 'M-VAD_full_movies':
            return 'M-VAD'
        else:
            return dir

    # Delete and recreate database
    if os.path.exists(VIDEOPATH_DB):
        os.remove(VIDEOPATH_DB)
    conn = sqlite3.connect(VIDEOPATH_DB)
    conn.execute('CREATE TABLE VideoPath('
                 'category TEXT,'
                 'title TEXT,'
                 'datasets TEXT,'
                 'dirpath TEXT,'
                 'movie_fn TEXT,'
                 'ext TEXT,'
                 'has_frames INTEGER,'
                 'num_frames INTEGER)')

    # Find all directories with movies
    with conn:
        cur = conn.cursor()
        for root, dirs, files in os.walk(VIDEOS_PATH):
            movie_fn = get_movie_fn_if_exists(files)
            if movie_fn:
                # root: data/videos/films/MovieQA_full_movies/Yes Man (2008)
                title = os.path.basename(root)
                category = root.split(VIDEOS_PATH)[1].split('/')[1]
                dirpath = root
                # print root
                datasets = get_dataset_name_from_dir(root.split(category)[1].split('/')[1])
                ext = movie_fn.split('.')[-1]
                has_frames = int(('frames' in dirs) and (len(os.listdir(os.path.join(root, 'frames'))) > 0))
                num_frames = len(os.listdir(os.path.join(root, 'frames'))) if has_frames else 0

                print category, title, datasets, dirpath, movie_fn, ext, has_frames, num_frames

                cur.execute("INSERT INTO VideoPath VALUES(?, ?, ?, ?, ?, ?, ?, ?)", (
                    category,
                    title.decode('utf8'),
                    datasets,
                    dirpath.decode('utf8'),
                    movie_fn.decode('utf8'),
                    ext,
                    has_frames,
                    num_frames
                ))

    # TODO and note on datasets field (not high priority, as datasets not being used right now)
    # 1) datasets is meant to track which existing datasets movie is also a part of.
    # For instance, the bulk of the movies are from the MovieQA and M-VAD datasets.
    # These were chosen in the case we wanted to do further analysis on these movies -- these datasets
    # provide extra DVS descriptions and metadata. (They also include large chunks of the movie, but not the full
    # movie, which is why they were downloaded in full, and hence the name of the directories, e.g. MovieQA_full_movies)
    # 2) However, there is some overlap between these datasets (as well as other datasets, such as CMU_movie_tropes).
    # datasets is supposed to be a comma-separated list of these datasets. In order to make it complete, I should
    # match the movies from the txt files containing the list of movies and upsert into the table.

########################################################################################################################
# VideoMetadata DB
########################################################################################################################
def match_film_metadata():
    """
    Get metadata for each film using CMU_movie_summary dataset

    Notes
    -----
    Assumes each title is similar to the format: <title> (<year>), e.g. King's Speech (2010)

    movie.metadata.tsv columns:
    # 1. Wikipedia movie ID ('975900')
    # 2. Freebase movie ID ('/m/03vyhn')
    # 3. Movie name ('Ghosts of Mars')
    # 4. Movie release date (2001-08-24)
    # 5. Movie box office revenue  ('14010832')
    # 6. Movie runtime ('98.0')
    # 7. Movie languages (Freebase ID:name tuples) ('{"/m/02h40lc": "English Language"}')
    # 8. Movie countries (Freebase ID:name tuples) ('{"/m/09c7w0": "United States of America"}'
    # 9. Movie genres (Freebase ID:name tuples) ('{"/m/01jfsb": "Thriller", "/m/06n90": "Science Fiction", ...}\n'}
    """

    # Get all metadata
    movie2metadata = {}
    with open(os.path.join(CMU_PATH, 'movie.metadata.tsv'), 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            movie2metadata[line[2]] = {'date': line[3],
                                       'revenue': None if line[4] == '' else int(line[4]),
                                       'runtime': None if line[5] == '' else float(line[5]),
                                       'genres': json.loads(line[8]).values()}
    movies = set(movie2metadata.keys())

    # Get tropes for each movie in tvtropes data
    movie2tropes = defaultdict(list)
    with open(os.path.join(CMU_PATH, 'tvtropes.clusters.txt'), 'r') as f:
        for line in f.readlines():
            trope, movie_data = line.split('\t')
            movie_data = json.loads(movie_data)     # char, movie, id, actor
            movie2tropes[movie_data['movie']].append(trope)

    # Write moves to file for the hell of it, so I can spot check
    with io.open('notes/CMU_tvtropes_movies.txt', 'w') as f:
        for movie in movie2tropes.keys():
            f.write(movie)
            f.write(u'\n')

    # Match videos to movies with metadata
    # Most are a good match, and there's only around 500 movies, so I manually cleaned it as follows:
    # I ran it once and saved stdout to a text file and just went through all the ones that aren't
    # This was done when M-VAD and MovieQA were added to MoviePath DB,
    #   and a few movies from other, CMU_tv_tropes, and animated
    # Then I just found all the ones that were incorrect or not there.
    # The biggest mismatches are for movies with sequels
    # The ones not there are mostly newer movies
    # The following was done manually
    manually_matched = {
        'The Hobbit The Desolation of Smaug': None,
        'Wanted': 'Wanted', # not Wanted 2
        'Cruel Intentions': 'Cruel Intentions',
        'Blue Jasmine': None,
        'Still Alice': None,
        'The Godfather Part I': 'The Godfather',
        'Twilight Saga': 'Twilight',
        'A Royal Night Out': None,
        'American Heist': None,
        'Gone Girl': None,
        'A Walk Among the Tombstones': None,
        '12 Years a Slave': None,
        'Avatar': 'Avatar',
        'Wild Things': 'Wild Things',
        'Oceans 11': "Ocean's Eleven",
        'Chappie': None,
        'Kung Fu Panda': 'Kung Fu Panda',
        'Her': None,
        'X-Men Days of Future Past': None,
        'How to Train Your Dragon 2': None,
        'Carol': 'Carol',
        'The Intouchables': 'Intouchables',
        '22 Jump Street': None,
        'Marley and Me': None,
        'Before I Go to Sleep': None,
        'Taken': 'Taken',
        '2 Guns': None,
        '3 Days to Kill': None,
        'The Butterfly Effect': 'The Butterfly Effect',
        'Short Term 12': None,
        'Elizabeth': 'Elizabeth',
        'American Psycho': 'American Psycho',
        'Men In Black': 'Men in Black',
        'This Is 40': None,
        'The Grand Budapest Hotel': None,
        'Zipper': None,
        'Mrs Doubtfire': 'Mrs. Doubtfire',
        'The Godfather Part 3': 'The Godfather Part III',
        'Bad Santa': 'Bad Santa',
        'Divergent': None,
        'The Hobbit The Battle of The Five Armies': None,
        'Cold in July': None,
        'Absolutely Anything': None,
        'Harry Potter And The Deathly Hallows Part 2': 'Harry Potter and the Deathly Hallows \xe2\x80\x93 Part 2',
        'A Walk in the Woods': None,
        'Back to the Future II': 'Back to the Future Part II',
        'I Robot': 'I, Robot',
        'About Time': None,
        '71': None,
        'X2 X-Men United': None,
        'Iron Man': 'Iron Man',
        'Captain America Civil War': None,
        'Shrek': 'Shrek',
        'Zootopia': None,
        'Big Hero 6': None,
        'The Wind Rises': None,
        'Bruno': 'Br\xc3\xbcno',
        'The Guilt Trip': None,
        'The Adventures of Tintin': None,
        'Coherence': None,
        'Finding Dory': None,
        'Kubo And The Two Strings': None,
        'The Secret Life of Pets': None,
        'The Boxtrolls': None,
        'The Good Dinosaur': None
    }

    result = {}

    conn = sqlite3.connect(VIDEOPATH_DB)
    with conn:
        cur = conn.cursor()
        rows = cur.execute("SELECT title FROM VideoPath WHERE category=='films'")
        for row in rows:
            orig_title = row[0]     # i.e. includes year, e.g. Serenity (2003)
            orig_title = orig_title.encode('utf-8')
            m = re.match(r'(.+) \(\d+\)?$', orig_title)
            title = m.group(1)

            if title in manually_matched:
                print title, 'TITLE IN MANUALLY MATCHED'
                match = manually_matched[title]
                if match:
                    result[orig_title] = movie2metadata[match]
                    continue
            else:
                matched = sorted([(fuzz.ratio(title, movie_name), movie_name) for movie_name in movies])
                matched = [t for s, t in matched[::-1][:10]]        # top 10
                print title, matched
                match = matched[0]
                result[orig_title] = movie2metadata[match]

                # print title, matched
    return result

def get_shorts_metadata():
    """
    Aggregate metadata for each video in data/videos/shorts

    Notes
    -----
    Each folder downloaded from Vimeo should have 1) <title>.info.json, 2) <title>.annotations.xml, and 3) <title>.description

    Returns
    -------
    short2metadata: dict
        - key is title (name of dir) This title is the common key with VIDEOPATH_DB
        - value is dictionary of metadata
    - The metadata dictionary is a subset of the info.json data and includes the description. For all the videos I
    spotchecked, annotatiosn is empty, so I'm skipping it for now.
    """
    short2metadata = {}

    def parse_info_file(fp):
        data = json.load(open(fp, 'r'))
        info = {
            'comment_count': data.get('comment_count'),
            'description': data.get('description'),
            'display_id': data.get('display_id'),  # not sure what's the diff with 'id' -- one video I checked they were the same
            'duration': data.get('duration'),
            'fps': data.get('fps'),
            'fulltitle': data.get('fulltitle'),
            'id': data.get('id'),
            'like_count': data.get('like_count'),
            'subtitles': data.get('subtitles'),
            'title': data.get('title'),
            'uploader': data.get('uploader'),
            'uploader_id': data.get('uploader_id'),
            'uploader_url': data.get('uploader_url'),
            'upload_date': data.get('upload_date'),
            'view_count': data.get('view_count')
        }
        return info

    conn = sqlite3.connect(VIDEOPATH_DB)
    with conn:
        cur = conn.cursor()
        rows = cur.execute("SELECT dirpath, title FROM VideoPath WHERE category=='shorts'")
        for row in rows:
            dirpath, title = row[0], row[1]
            # print title
            info_fp = os.path.join(dirpath, title +  u'.info.json')
            info = parse_info_file(info_fp)
            short2metadata[title] = info
    return short2metadata

def create_videometadata_db():
    """
    Create VideoMetadata DB -- right now it's just a pkl file (I was looking for a lightweight no SQL database.)
    The keys are titles (common key with VideoPath DB).
    """
    db = {}

    print 'Getting shorts metadata'
    short2metadata = get_shorts_metadata()
    for title, metadata in short2metadata.items():
        db[title] = metadata

    print 'Getting films metadata'
    film2metadata = match_film_metadata()
    for title, metadata in film2metadata.items():
        db[title] = metadata

    print 'Saving'
    with open(VIDEOMETADATA_DB, 'w') as f:
        pickle.dump(db, f, protocol=2)

    print 'Done'

if __name__ == '__main__':

    # Set up commmand line arguments
    parser = argparse.ArgumentParser(description='Download and process data')
    parser.add_argument('--save_video_frames', dest='save_video_frames', action='store_true')
    parser.add_argument('--save_video_frames_sr', dest='save_video_frames_sr', type=float, default=1)
    parser.add_argument('--convert_avis_to_mp4s', dest='convert_avis_to_mp4s', action='store_true')
    parser.add_argument('--save_credits_index', dest='save_credits_index', action='store_true')
    parser.add_argument('--save_credits_index_overwrite', dest='save_credits_index_overwrite', default=False,
                        action='store_true', help='overwrite credits_index.txt files')
    parser.add_argument('--extract_highlight_clips', dest='extract_highlight_clips', action='store_true')
    parser.add_argument('--overwrite_clips', dest='overwrite_clips', action='store_true')
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--create_videopath_db', dest='create_videopath_db', action='store_true')
    parser.add_argument('--match_film_metadata', dest='match_film_metadata', action='store_true')
    parser.add_argument('--get_shorts_metadata', dest='get_shorts_metadata', action='store_true')
    parser.add_argument('--create_videometadata_db', dest='create_videometadata_db', action='store_true')
    parser.add_argument('--vids_dir', dest='vids_dir', default=None,
                        help='folder that contains dirs (one movie each), e.g. films/MovieQA_full_movies')
    parser.add_argument('--vids_dirpath', dest='vids_dirpath', default=None,
                        help='folder that contains movies, e.g. data/videos/films/')

    cmdline = parser.parse_args()

    if cmdline.save_video_frames:
        save_video_frames(cmdline.vids_dir, cmdline.save_video_frames_sr)
    elif cmdline.convert_avis_to_mp4s:
        convert_avis_to_mp4s(cmdline.vids_dir)
    elif cmdline.save_credits_index:
        save_credits_index(cmdline.vids_dir, overwrite_files=cmdline.save_credits_index_overwrite)
    elif cmdline.extract_highlight_clips:
        extract_highlight_clips(cmdline.vids_dirpath, cmdline.overwrite_clips, cmdline.verbose)
    elif cmdline.create_videopath_db:
        create_videopath_db()
    elif cmdline.match_film_metadata:
        pprint(match_film_metadata())
    elif cmdline.get_shorts_metadata:
        pprint(get_shorts_metadata())
    elif cmdline.create_videometadata_db:
        create_videometadata_db()