# Prepare the data (parse, convert to tfrecords, download, etc.)

import argparse
import cPickle as pickle
from itertools import chain
import librosa
import matplotlib.pylab as plt
from multiprocessing.dummy import Pool as ThreadPool
from natsort import natsorted
import numpy as np
import os
from pprint import pprint
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import sqlite3
import string
import subprocess
import tensorflow as tf
import urllib

from core.utils.utils import VID_EXTS

MSD_FILES_PATH = 'data/msd/full/AdditionalFiles/'
MSD_TRACKS_DB = 'data/msd/full/AdditionalFiles/track_metadata.db'
MSD_SPOTIFY_MATCH_RESULTS  = 'data/msd/full/AdditionalFiles/msd_spotify_match_results_{}.pkl'

SPOTIFY_PATH = 'data/spotify/'
MSD_SPOTIFY_MATCH_DB = 'data/spotify/MsdSpotifyMatch.db'
SPOTIFY_AUDIO_FEATURES = 'data/spotify/spotify_audio_features.pkl'
SPOTIFY_AF_DB = 'data/spotify/SpotifyAudioFeatures.db'
SPOTIFY_PREVIEWS_PATH = 'data/spotify/previews/'
SONGS_PER_FOLDER = 100   # 26 * 26 * 26 * 75 = 1757600

MELGRAM_30S_SIZE = (96, 1407)
NUMPTS_AND_MEANSTD_PATH = 'data/spotify/numpts_and_meanstd.pkl'

# Mel-spectrogram parameters
N_FFT = 512
N_MELS = 96
HOP_LEN = N_FFT / 2   # overlap 50%

########################################################################################################################
# Million song dataset (msd) and Spotify
# Overview of dataset created and relationship between msd and Spotify:
# 1) match_msd_with_spotify(): search song and artist names in msd on Spotify
#       - msd_track_id, spotify_track_id, and spotify_preview_url are saved in MSD_SPOTIFY_MATCH_DB
# 2) lookup_spotify_audio_features(): for all songs in above MSD_SPOTIFY_MATCH_DB with both spotify fields, lookup
# the spotify audio features, which contain labels such as valence and energy.
#       - This requires app authentication. before running this function, run the following in the command line:
#           - export SPOTIPY_CLIENT_ID='your-spotify-client-id'
#           - export SPOTIPY_CLIENT_SECRET='your-spotify-client-secret'
# 3) dl_previews_from_spotify(): for all songs that have audio features looked up, obtain the 30 second preview
# from the preview_url.
########################################################################################################################

########################################################################################################################
# Download and clean MSD-Spotify data
########################################################################################################################
def download_msd_previews_from_7digital():
    # Use https://github.com/mlachmish/MusicGenreClassification
    # NOTE: very weird bug - doesn't work if country is US or GB, works if it's ww
    # Need to have created 7digital api account
    # Rate limited to 4000 requests per day
    pass

def get_all_msd_songs_info():
    """
    Return dict mapping msd track_id to dict with identifying information about the song. Reads from msd
    track_metadata.db.
    """
    track_id2info = {}

    conn = sqlite3.connect(MSD_TRACKS_DB)
    cur = conn.cursor()
    rows = cur.execute("SELECT track_id, title, artist_name, year, duration FROM songs")
    for row in rows:
        track_id2info[row[0]] = {
            'title': row[1],
            'artist_name': row[2],
            'year': row[3],
            'duration': row[4]
        }

    return track_id2info

def get_all_match_results():
    """
    If implemented, return all [msd_track_ids, spotify_track_ids], etc from the MSD_SPOTIFY_MATCH_DB
    """
    pass

def match_msd_with_spotify():
    """
    For each track in msd, search on Spotify. Save all the html responses into pickle files. For all responses,
    save msd_track_id, spotify_track_id, and spotify_preview_url into MSD_SPOTIFY_MATCH_DB. For some responses,
    the spotify fields will not exist.

    Details
    -------
    - Example api url: 'https://api.spotify.com/v1/search?q=track:Positive+Balance%20artist:Immortal+Technique&type=track'
    - Since we are saving the entire response for a million songs, saving it all in one pickle would be quite big (on
    the order of 10 GB). Thus, they are saved in msd_spotify_match_results_{}.pkl, where {} is 0,1,2,.... Each pickle
    file contains responses for 100,000 tracks.
    - Use the first result from the search.
    """

    # Make db if it doesn't exist
    if not os.path.exists(MSD_SPOTIFY_MATCH_DB):
        conn = sqlite3.connect(MSD_SPOTIFY_MATCH_DB)
        conn.execute('CREATE TABLE MsdSpotifyMatch('
                     'msd_track_id TEXT,'
                     'spotify_track_id TEXT,'
                     'spotify_preview_url TEXT)')

    # Get msd tracks already matched
    conn = sqlite3.connect(MSD_SPOTIFY_MATCH_DB)
    cur = conn.cursor()
    rows = cur.execute("SELECT msd_track_id FROM MsdSpotifyMatch")
    already_matched = set()
    for row in rows:
        already_matched.add(row[0])

    # Get current match results file (which contains full responses)
    cur_msmr_fns = natsorted([f for f in os.listdir(MSD_FILES_PATH) if f.startswith('msd_spotify_match_results')])
    cur_msmr_fn = cur_msmr_fns[-1]
    cur_msmr_idx = len(cur_msmr_fns) - 1
    cur_msmr = pickle.load(open(os.path.join(MSD_FILES_PATH, cur_msmr_fn), 'rb'))

    request_str = u'https://api.spotify.com/v1/search?q=track:{}%20artist:{}&type=track'
    track_id2info = get_all_msd_songs_info()

    print 'Matching with spotify'
    i = 0
    for msd_track_id, info in track_id2info.items():
        if msd_track_id in already_matched:
            print 'Track already looked up, skipping'
            continue

        song = info['title']
        song_str = '+'.join(song.split())
        artist = info['artist_name']
        artist_str = '+'.join(artist.split())

        # print song, artist, msd_track_id
        r = requests.get(request_str.format(song_str, artist_str))
        cur_msmr[msd_track_id] = r

        try:
            j = r.json()
            spotify_track_id = j['tracks']['items'][0]['id']
            preview_url = j['tracks']['items'][0]['preview_url']
            # print msd_track_id, spotify_track_id, preview_url

            cur.execute("INSERT INTO MsdSpotifyMatch VALUES(?, ?, ?)", (
                    msd_track_id,
                    spotify_track_id,
                    preview_url))

        except Exception as e:
            # print e
            cur.execute("INSERT INTO MsdSpotifyMatch VALUES(?, ?, ?)", (
                    msd_track_id,
                    None,
                    None))

        # Add to db
        if i % 100 == 0:
            print '=' * 100
            print 'Commiting to db'
            print '=' * 100
            conn.commit()

        # Add responses to pickle
        # Create new file every 100,000
        if len(cur_msmr) % 1000 == 0:
            print '=' * 100
            print 'Saving pickle'
            print '=' * 100
            msmr_out_fp = os.path.join(MSD_SPOTIFY_MATCH_RESULTS.format(cur_msmr_idx))
            with open(msmr_out_fp, 'wb') as f:
                pickle.dump(cur_msmr, f, protocol=2)

            if len(cur_msmr) == 100000:
                cur_msmr_idx += 1
                cur_msmr = {}

        i += 1

    # Close up
    conn.commit()
    msmr_out_fp = os.path.join(MSD_SPOTIFY_MATCH_RESULTS.format(cur_msmr_idx))
    with open(msmr_out_fp, 'wb') as f:
        pickle.dump(cur_msmr, f, protocol=2)

# def clean_msd_spotify_matched():
#     """
#     Remove all tracks where no 'tracks' in response, or maybe response timed out, etc., unicode in string
#     """
#     matched_so_far = pickle.load(open(SPOTIFY_MATCH_RESULTS, 'rb'))  # dict: key is msd track_id, value is request result
#
#     for track_id, response in matched_so_far.items():
#         try:
#             j = response.json()
#         except Exception as e:
#             print track_id, e
#
#         if 'tracks' not in j:
#             print track_id

def lookup_spotify_audio_features():
    """
    For all songs in MSD_SPOTIFY_MATCH_DB with both spotify fields, lookup the spotify audio features, which contain
    labels such as valence and energy.
    """
    print 'Looking up Spotify audio features'

    # Get previously looked up features
    lookedup_so_far = pickle.load(open(SPOTIFY_AUDIO_FEATURES, 'rb'))   # dict: key is spotify track_id, value is audio features

    # Get spotify_track_ids that have been matched with msd
    conn = sqlite3.connect(MSD_SPOTIFY_MATCH_DB)
    cur = conn.cursor()
    all_spotify_track_ids = set()
    rows = cur.execute("SELECT spotify_track_id FROM MsdSpotifyMatch "
                       "WHERE (spotify_track_id is not NULL) AND (spotify_preview_url is not NULL)")
    for row in rows:
        all_spotify_track_ids.add(row[0])
    all_spotify_track_ids = list(all_spotify_track_ids)

    # Looking up audio features requires authentication
    client_credentials_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    cur_batch_spotify_track_ids = []        # make a request of 50 ids at a time
    i = 0                                   # number batches looked up this run, just used for printing
    for spotify_track_id in all_spotify_track_ids:
        if spotify_track_id in lookedup_so_far:
            # print 'Track already looked up, skipping'
            continue
        else:
            cur_batch_spotify_track_ids.append(spotify_track_id)


        # Lookup and save features in batches
        if len(cur_batch_spotify_track_ids) % 50 == 0:
            tracks_features = sp.audio_features(cur_batch_spotify_track_ids)
            for j, track_features in enumerate(tracks_features):
                lookedup_so_far[cur_batch_spotify_track_ids[j]] = track_features

            # Reset
            cur_batch_spotify_track_ids = []

        # Save pickle
        if i % 1000 == 0:
            print '=' * 100
            print 'Pickling'
            with open(SPOTIFY_AUDIO_FEATURES, 'wb') as f:
                pickle.dump(lookedup_so_far, f, protocol=2)
            # Print a count
            print 'cur_run: {}, looked_up_so_far: {}, num_match: {}'.format(
                i, len(lookedup_so_far), len(all_spotify_track_ids))
            print '=' * 100

        i += 1

    # Wrap up
    with open(SPOTIFY_AUDIO_FEATURES, 'wb') as f:
        pickle.dump(lookedup_so_far, f, protocol=2)

def clean_spotify_audio_features():
    """
    Basic checking of the audio features. Vast majority so far are 'good' and have the valence and energy fields:
    277895 good, 654 bad
    """
    with open(SPOTIFY_AUDIO_FEATURES, 'rb') as f:
        id2af = pickle.load(f)

    bad = 0
    good = 0
    v, e = 0.0, 0.0
    for id, af in id2af.items():
        if af == None:
            # print id
            bad += 1
            continue
        if ('valence' not in af.keys()) or ('energy' not in af.keys()) or \
                (af['valence'] is None) or (af['energy'] is None):
            bad += 1
            # print id
            continue

        v += af['valence']
        e += af['energy']
        good += 1
    print v/good
    print e/good
    print good
    print bad

def put_spotify_af_in_db():
    """
    Place the entries in the spotify_audio_feaures.pkl into a sqlite db
    """
    print 'Putting Spotify audio features in sqlite db'

    # Make db if it doesn't exist
    if not os.path.exists(SPOTIFY_AF_DB):
        conn = sqlite3.connect(SPOTIFY_AF_DB)
        conn.execute('CREATE TABLE SpotifyAudioFeatures('
                     'acousticness REAL,'
                     'analysis_url TEXT,'
                     'danceability REAL,'
                     'duration_ms REAL,'
                     'energy REAL,'
                     'id TEXT,'
                     'instrumentalness REAL,'
                     'key INTEGER,'
                     'liveness REAL,'
                     'loudness REAL,'
                     'mode INTEGER,'
                     'speechiness REAL,'
                     'tempo REAL,'
                     'time_signature INTEGER,'
                     'track_href TEXT,'
                     'type TEXT,'
                     'uri TEXT,'
                     'valence REAL)')

    conn = sqlite3.connect(SPOTIFY_AF_DB)
    cur = conn.cursor()

    af = pickle.load(open(SPOTIFY_AUDIO_FEATURES, 'rb'))
    for spotify_track_id, features in af.items():
        if features:
            cur.execute("INSERT INTO SpotifyAudioFeatures VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                        features.get('acousticness'),
                        features.get('analysis_url'),
                        features.get('danceability'),
                        features.get('duration_ms'),
                        features.get('energy'),
                        features.get('id'),
                        features.get('instrumentalness'),
                        features.get('key'),
                        features.get('liveness'),
                        features.get('loudness'),
                        features.get('mode'),
                        features.get('speechiness'),
                        features.get('tempo'),
                        features.get('time_signature'),
                        features.get('track_href'),
                        features.get('type'),
                        features.get('uri'),
                        features.get('valence')))
        else:
            print spotify_track_id

    conn.commit()

def plot_af_stats():
    # Get histogram of valence and energy
    plt.style.use('ggplot')

    conn = sqlite3.connect(SPOTIFY_AF_DB)
    cur = conn.cursor()
    rows = cur.execute("SELECT valence, energy from SpotifyAudioFeatures")
    valences, energies, valergies = [], [], []
    for row in rows:
        if row[0] and row[1]:
            valences.append(row[0])
            energies.append(row[1])
        # if row[1]:
        # if row[0] and row[1]:
        #     valergies.append([row[0], row[1]])
    plt.hist(valences, 25)
    plt.show()
    plt.hist(energies, 25)
    plt.show()
    plt.hexbin(valences, energies, gridsize=25)#, bins='log', cmap='inferno')
    plt.show()


def dl_previews_from_spotify():
    """
    Download previews from spotify urls and not using 7preview. For all the songs a) matched on MSD, and b)
    have Spotify audio features looked up, use the preview_url obtained while matching to download the preview.
    Save songs in */*/*/, where * is an alphabet letter, and there are 100 songs in each leaf directory. When running
    after some files have already been downloaded, figure out which dir to save files by reading
    previews_dir_indices.txt. This contains 4 numbers - the first 3 correspond to the index of the alphabet letter
    of the directories, and the 4th is the count of how many files are in the leaf dir currently. For instance,
    if B/E/Z/ is the last folder saved to and currently contains 53 mp3 files, the 4 numbers should be 1,4,25,53.
    """
    def retrieve_img_and_process(url_and_fp):
        url, fp = url_and_fp[0], url_and_fp[1]
        urllib.urlretrieve(url, fp)

    # Get files already downloaded
    already_downloaded = set()
    num_already_downloaded = 0
    # Note: num_already_downloaded may be different from len(already_downloaded). Maybe because two different msd
    # searches produced same spotify result
    # But not a big deal because as long as we have the spotify preview and features, we're good.
    # Just need to write a function to remove duplicates after, even if that means not every folder has 100 songs
    for root, dirs, files in os.walk(SPOTIFY_PREVIEWS_PATH):
        if len(dirs) == 0:          # leaf directories
            for f in files:
                already_downloaded.add(f.split('.')[0])
            num_already_downloaded += len(files)
    print 'Already downloaded: {} tracks'.format(num_already_downloaded)

    # Used for filepaths
    alph = string.ascii_uppercase
    dir_indices = open(os.path.join(SPOTIFY_PATH, 'previews_dir_indices.txt'), 'rb').readlines()[0].split(',')
    dir_indices = [int(s) for s in dir_indices]
    cur_alph_indices = dir_indices[0:3]
    i = dir_indices[3]
    print cur_alph_indices, i

    # Check that indices are correct
    num_files_idx = (SONGS_PER_FOLDER * cur_alph_indices[2]) + \
            (SONGS_PER_FOLDER * (cur_alph_indices[1] * 26)) + \
            (SONGS_PER_FOLDER * (cur_alph_indices[0] * 26 * 26)) + \
            i   # i is supposed to be index for next file to save, so if A/Z/B/ has one file, i = 1
    if num_files_idx != num_already_downloaded:
        print 'Indices incorrect: {} vs. {}'.format(num_files_idx, num_already_downloaded)
        return


    # Get spotify_track_ids and preview urls that have been matched with msd
    conn = sqlite3.connect(MSD_SPOTIFY_MATCH_DB)
    cur = conn.cursor()
    spotify_track_ids_and_preview_urls = []
    rows = cur.execute("SELECT spotify_track_id, spotify_preview_url FROM MsdSpotifyMatch "
                       "WHERE (spotify_track_id is not NULL) AND (spotify_preview_url is not NULL)")
    for row in rows:
        spotify_track_ids_and_preview_urls.append([row[0], row[1]])

    # Get spotify tracks that have had audio features looked up
    af = pickle.load(open(SPOTIFY_AUDIO_FEATURES, 'rb'))   # dict: key is spotify track_id, value is audio features

    print 'Getting urls and fps'
    urls = []
    fps = []
    for spotify_track_id, preview_url in spotify_track_ids_and_preview_urls:

        # Skip if preview mp3 already downloaded
        if spotify_track_id in already_downloaded:
            continue

        # Skip if don't have audio features yet
        if spotify_track_id not in af:
            continue

        urls.append(preview_url)

        # Get fiilepath
        dir = alph[cur_alph_indices[0]] + '/' + alph[cur_alph_indices[1]] + '/' + alph[cur_alph_indices[2]]
        dirpath = os.path.join(SPOTIFY_PREVIEWS_PATH, dir)
        if not os.path.exists(dirpath):
            os.system('mkdir -p {}'.format(dirpath))
        fp = os.path.join(dirpath, '{}.mp3'.format(spotify_track_id))
        fps.append(fp)
        # print fp

        # Update fp indices
        if (i != 0) and (i % SONGS_PER_FOLDER == 0):
            cur_alph_indices[2] += 1
            if cur_alph_indices[2] > 25:
                cur_alph_indices[2] = 0
                cur_alph_indices[1] += 1
            if cur_alph_indices[1] > 25:
                cur_alph_indices[2] = 0
                cur_alph_indices[1] = 0
                cur_alph_indices[0] += 1

        i += 1

    print 'Downloading previews for {} tracks'.format(len(urls))
    pool = ThreadPool(100)
    urls_and_fps = zip(urls, fps)
    step = 1000
    print len(urls)
    print len(fps)
    print len(urls_and_fps)

    for i in range(0, len(urls_and_fps), step):
        print i
        pool.map(retrieve_img_and_process, urls_and_fps[i:i+step])

def clean_dld_previews():
    """
    Remove duplicates, make all leaf nodes have 100 files
    """
    # Remove duplicates
    dups = 0
    already_downloaded = set()
    for root, dirs, files in os.walk(SPOTIFY_PREVIEWS_PATH):
        if len(dirs) == 0:
            for f in files:
                if f in already_downloaded:
                    os.remove(os.path.join(root, f))
                    dups += 1
                else:
                    already_downloaded.add(f)
    print 'Removed {} duplicates'.format(dups)

    # # Make every leaf folder (except the last one maybe) have 100 files
    # This is pretty rare and easy to do more manually through Python terminal
    # Get last folder to move
    alph = string.ascii_uppercase
    dir_indices = open(os.path.join(SPOTIFY_PATH, 'previews_dir_indices.txt'), 'rb').readlines()[0].split(',')
    dir_indices = [int(s) for s in dir_indices]
    cur_alph_indices  = dir_indices[0:3]
    print cur_alph_indices

    for root, dirs, files in os.walk(SPOTIFY_PREVIEWS_PATH):
        last_dir = alph[cur_alph_indices[0]] + '/' + alph[cur_alph_indices[1]] + '/' + alph[cur_alph_indices[2]]
        last_dirpath = os.path.join(SPOTIFY_PREVIEWS_PATH, last_dir)
        if len(dirs) == 0 and len(files) != SONGS_PER_FOLDER and root != last_dirpath:
            diff = SONGS_PER_FOLDER - len(files)
            if diff < 0:        # move files out of this folder and into the last folder
                for i in range(diff):
                    src_path = os.path.join(SPOTIFY_PREVIEWS_PATH, root, files[i])
                    last_path = os.path.join(SPOTIFY_PREVIEWS_PATH, last_dir, files[i])
                    print src_path, last_path
                    # os.rename(src_path, last_path)
            elif diff > 0:      # move files from the last folder and into this folder
                for i in range(diff):
                    src_dir = alph[cur_alph_indices[0]] + '/' + alph[cur_alph_indices[1]] + '/' + alph[cur_alph_indices[2]]
                    src_dirpath = os.path.join(SPOTIFY_PREVIEWS_PATH, src_dir)
                    src_files = os.listdir(src_dirpath)
                    # print src_dirpath, len(src_files)

                    # Possibly update src folder if no more files by going backwards (D/F/Z -> D/F/Y)
                    if len(src_files) == 0:
                        cur_alph_indices[2] -= 1
                        if cur_alph_indices[2] < 0:
                            cur_alph_indices[2] = 25
                            cur_alph_indices[1] -= 1
                        if cur_alph_indices[1] < 0:
                            cur_alph_indices[2] = 25
                            cur_alph_indices[1] = 25
                            cur_alph_indices[0] -= 1
                        src_dir = alph[cur_alph_indices[0]] + '/' + alph[cur_alph_indices[1]] + '/' + alph[cur_alph_indices[2]]
                        src_dirpath = os.path.join(SPOTIFY_PREVIEWS_PATH, src_dir)
                        src_files = os.listdir(src_dirpath)

                    src_fp = os.path.join(src_dirpath, src_files[0])
                    dest_fp = os.path.join(root, src_files[0])
                    print src_fp, dest_fp, len(src_files)
                    os.rename(src_fp, dest_fp)

########################################################################################################################
# Turn cleaned spotify data into tfrecords for training
########################################################################################################################
def write_spotify_to_tfrecords(split=[0.8, 0.1, 0.1]):
    """
    Create tfrecord file for each biconcept for train,valid,test
    """
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    # Make directory for tfrecords - train, valid, test
    for name in ['train', 'valid', 'test']:
        path = os.path.join(SPOTIFY_PATH, 'tfrecords', name)
        if not os.path.exists(path):
            os.system('mkdir -p {}'.format(path))

    # Load audio features
    af = pickle.load(open(SPOTIFY_AUDIO_FEATURES, 'rb'))

    for root, dirs, files in os.walk(SPOTIFY_PREVIEWS_PATH):
        # Group tfrecords at the second level, i.e. AB.tfrecord will have all the mp3's from A/B/*
        # 2600 seems like a reasonable size
        alphs = root.split(SPOTIFY_PREVIEWS_PATH)[1].replace('/','')        # eg. A, AB, AXY
        if len(alphs) == 2:
            print alphs

            fps = [[os.path.join(root, d, f) for f in os.listdir(os.path.join(root, d)) if not f.startswith('.')] for d in dirs]
            fps = list(chain(*fps))     # flatten

            # Get tfrecord filepath and writer ready
            tfrecords_filename = '{}.tfrecords'.format(alphs)
            tr_tfrecords_fp = os.path.join(SPOTIFY_PATH, 'tfrecords', 'train', tfrecords_filename)

            if os.path.exists(tr_tfrecords_fp):
                print '{} exists, skipping'.format(tr_tfrecords_fp)
                continue

            va_tfrecords_fp = os.path.join(SPOTIFY_PATH, 'tfrecords', 'valid', tfrecords_filename)
            te_tfrecords_fp = os.path.join(SPOTIFY_PATH, 'tfrecords', 'test', tfrecords_filename)
            tr_writer = tf.python_io.TFRecordWriter(tr_tfrecords_fp)
            va_writer = tf.python_io.TFRecordWriter(va_tfrecords_fp)
            te_writer = tf.python_io.TFRecordWriter(te_tfrecords_fp)
            train_endidx = int(split[0] * len(fps))
            valid_endidx = train_endidx + int(split[1] * len(fps))

            # Convert images to tfrecord examples
            for i, fp in enumerate(fps):
                try:
                    track_id = os.path.basename(fp).split('.')[0]
                    tfrecord_ex_id = fp.split('/')[3] + fp.split('/')[4] + fp.split('/')[5] + '-' + track_id
                    log_melgram = compute_log_melgram(fp)
                    log_melgram = log_melgram.astype(np.float32, copy=False)        # float64 -> float32
                    log_melgram_raw = log_melgram.tostring()
                    print i, tfrecord_ex_id

                    # print np.fromstring(log_melgram_raw).shape

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'id': _bytes_feature(tfrecord_ex_id),
                        'log_melgram': _bytes_feature(log_melgram_raw),
                        'valence_reg': _float_feature(af[track_id]['valence']),
                        'energy_reg': _float_feature(af[track_id]['energy']),
                        'tempo_reg': _float_feature(af[track_id]['tempo']),
                        'speechiness_reg': _float_feature(af[track_id]['speechiness']),
                        'danceability_reg': _float_feature(af[track_id]['danceability']),
                        'key_reg': _float_feature(af[track_id]['key']),
                        'loudness_reg': _float_feature(af[track_id]['loudness'])
                    }))

                    # Figure out which writer to use (train, valid, test)
                    if i < train_endidx:
                        writer = tr_writer
                    elif i >= train_endidx and i < valid_endidx:
                        writer = va_writer
                    else:
                        writer = te_writer

                    writer.write(example.SerializeToString())

                except Exception as e:
                    print fp, e

    tr_writer.close()
    va_writer.close()
    te_writer.close()

def compute_log_melgram(audio_path):
    """
    Compute a mel-spectrogram and return a np array of shape (96,1407), where
    96 == #mel-bins and 1407 == #time frame
    """

    # Audio and mel-spectrogram parameters
    SR = 12000
    DUR = 30              # in seconds

    # Load audio and downsample
    src, orig_sr = librosa.load(audio_path, sr=None)  # whole signal at native sampling rate
    src = librosa.core.resample(src, orig_sr, SR)     # downsample down to SR
    melgram = compute_log_melgram_from_np(src, DUR, SR, HOP_LEN, N_FFT, N_MELS)

    return melgram

def compute_log_melgram_from_np(src, dur, sr, hop_len, n_fft, n_mels):
    """
    Compute a mel-spectrogram from a numpy array
    """
    # Adjust size if necessary. Vast, vast majority of mp3's are 30 seconds and should require little adjustment.
    n_sample = src.shape[0]
    n_sample_fit = int(dur * sr)
    if n_sample < n_sample_fit:                       # if too short, pad with zeros
        src = np.hstack((src, np.zeros((int(dur * sr) - n_sample,))))
    elif n_sample > n_sample_fit:                     # if too long, take middle section of length DURA seconds
        src = src[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]

    # Compute log mel spectrogram
    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y =src, sr=sr, hop_length=hop_len,
                        n_fft=n_fft, n_mels=n_mels)**2,
                ref_power=1.0)
#     ret = ret[np.newaxis, np.newaxis, :]
    return ret

def precompute_numpts_and_meanstd_from_tfrecords():
    """
    From the tfrecords, calculate a) number of pts per train-valid-test split, and b) the per mel-bin mean and
    stddev. This will be loaded and used in datasets.py.
    """
    # Add counts. TODO: this is slow (not a huge deal considering it's a one-time setup), but still...
    numpts_and_meanstd = {}
    numpts_and_meanstd['num_pts'] = {}
    for split in ['train', 'valid', 'test']:
        n = 0
        mean = np.zeros(MELGRAM_30S_SIZE[0])        # (num mel-bins, )
        std = np.zeros(MELGRAM_30S_SIZE[0])         # (num mel-bins, )
        dirpath = os.path.join(SPOTIFY_PATH, 'tfrecords', split)
        for tfrecord in os.listdir(dirpath):
            # if tfrecord.startswith('HP'):          # Used to test while still saving tfrecords
            #     continue

            tfrecord_path = os.path.join(dirpath, tfrecord)
            for record in tf.python_io.tf_record_iterator(tfrecord_path):
                n += 1
                if split == 'train':                # only calculate mean and std on train set
                    # print n, tfrecord
                    example = tf.train.Example()
                    example.ParseFromString(record)

                    log_melgram_str = (example.features.feature['log_melgram'].bytes_list.value[0])
                    log_melgram = np.fromstring(log_melgram_str, dtype=np.float32)
                    # print log_melgram.shape[0] / 96
                    log_melgram = log_melgram.reshape(MELGRAM_30S_SIZE)

                    # Update moving average of mean and std
                    # New average = old average * (n-1)/n + new value /n
                    mean = mean * (n-1)/n + log_melgram.mean(axis=1) / n
                    std = std * (n-1)/n + log_melgram.std(axis=1) / n

        numpts_and_meanstd['num_pts'][split] = n

        if split == 'train':
            mean = np.expand_dims(mean, 1)      # (num mel-bins, ) -> (num mel-bins, 1)
            std = np.expand_dims(std, 1)        # (num mel-bins, ) -> (num mel-bins, 1)
            numpts_and_meanstd['mean'] = mean
            numpts_and_meanstd['std'] = std

    # Save
    with open(NUMPTS_AND_MEANSTD_PATH, 'wb') as f:
        pickle.dump(numpts_and_meanstd, f, protocol=2)

    print numpts_and_meanstd['num_pts']

def modify_tfrecords():
    """
    Load and modify -- (a) turn melgrams from float64 to float32, (b) add more labels -- existing tfrecords.
    Used because initial run creating tfrecords used float64 and only saved valence_reg. Modifying instead of
    re-creating tfrecords because creating tfrecords is slow due to loading each mp3 file and calculating
    melgram.
    """
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def check_features_not_none(track_af, feature_names):
        for feat in feature_names:
            if (feat not in track_af) or (track_af[feat] is None):
                return False
        return True

    # Load audio features
    print 'Loading audio features'
    af = pickle.load(open(SPOTIFY_AUDIO_FEATURES, 'rb'))

    for split in ['train', 'valid', 'test']:
        dirpath = os.path.join(SPOTIFY_PATH, 'tfrecords', split)
        for tfrecord in os.listdir(dirpath):
            print '{} -- {}'.format(split, tfrecord)
            tfrecord_path = os.path.join(dirpath, tfrecord)

            writer = tf.python_io.TFRecordWriter(tfrecord_path + '.modified')
            for record in tf.python_io.tf_record_iterator(tfrecord_path):
                example = tf.train.Example()
                example.ParseFromString(record)

                # Get log melgram and convert to float32
                log_melgram_str = (example.features.feature['log_melgram'].bytes_list.value[0])
                log_melgram = np.fromstring(log_melgram_str)
                log_melgram = log_melgram.astype(np.float32, copy=False)
                log_melgram_str = log_melgram.tostring()

                # print log_melgram.shape[0] / 96

                # Get id to get more audio features
                track_id = example.features.feature['id'].bytes_list.value[0].split('-')[1]
                cur_af = af[track_id]
                valid = check_features_not_none(cur_af, ['energy', 'tempo', 'speechiness', 'danceability', 'key', 'loudness'])
                if not valid:
                    print '{} missing features, skipping'.format(track_id)
                    continue

                modified = tf.train.Example(features=tf.train.Features(feature={
                    'id': _bytes_feature(example.features.feature['id'].bytes_list.value[0]),
                    'log_melgram': _bytes_feature(log_melgram_str),
                    'valence_reg': _float_feature(example.features.feature['valence_reg'].float_list.value[0]),
                    'energy_reg': _float_feature(cur_af['energy']),
                    'tempo_reg': _float_feature(cur_af['tempo']),
                    'speechiness_reg': _float_feature(cur_af['speechiness']),
                    'danceability_reg': _float_feature(cur_af['danceability']),
                    'key_reg': _float_feature(cur_af['key']),
                    'loudness_reg': _float_feature(cur_af['loudness']),
                }))

                writer.write(modified.SerializeToString())

            # Replace old with modified
            os.system('rm {}'.format(tfrecord_path))
            os.system('mv {} {}'.format(tfrecord_path + '.modified', tfrecord_path))

########################################################################################################################
# Extract audio for videos in order to predict audio-based emotional curves
########################################################################################################################
def extract_audio_from_vids(vids_dirpath):
    """
    Save mp3's of audio for every video in vids_dirpath
    """
    errors_f = open('notes/audio_extraction_errors.txt', 'wb')
    for root, dirs, files in os.walk(vids_dirpath):
        # See if audio exists already, skip if it does

        # Find movie fn
        movie_fn = None
        for f in files:
            for ext in VID_EXTS:
                if f.endswith(ext):
                    movie_fn = f
                    break

        if movie_fn:
            try:
                movie_fp = os.path.join(root, movie_fn)
                out_fn = movie_fn.split('.')
                out_fn = '.'.join(out_fn[0:len(out_fn)-1]) + '.mp3'
                out_fp = os.path.join(root, out_fn)

                # Skip if audio file already exists
                if os.path.exists(out_fp):
                    print 'Audio already exists for: {}, skipping'.format(out_fp)
                    continue

                print '=' * 100
                print '=' * 100
                print '=' * 100
                print 'Extracting audio for: {}'.format(movie_fp)
                print 'Saving to: {}'.format(out_fp)
                # NOTE: 12000 sample rate because that's what used for tfrecord melgram features
                # Note: Not using split() on bash command string because path's may have spaces
                cmd = ['ffmpeg', '-i'] + [movie_fp] + ['-ar', '12000', '-q:a', '0', '-map', 'a'] + [out_fp]
                subprocess.call(cmd, stdout=subprocess.PIPE)
                print 'Done extracting'

            except Exception as e:
                print movie_fn, e
                errors_f.write(u'{},{}\n'.format(unicode(movie_fn, 'utf-8'), unicode(e, 'utf-8')))

    errors_f.close()

def remove_bad_mp3s(vids_dirpath, size):
    """
    Remove all mp3's that are less than size in MB. Should be relatively easy to remove bad mp3s for movies by setting
    size to say 1, but not sure what the cutoff should be for shorts (0 will catch most but not all).
    Will probably have to come up with a different way to detect bad mp3s, e.g. length of mp3 should be relatively
    equal to length of frames/, length of movie.
    """
    for root, dirs, files in os.walk(vids_dirpath):
        for f in files:
            if f.endswith('mp3'):
                fp = os.path.join(root, f)
                nbytes = os.path.getsize(fp)
                nmb = nbytes / 1000000.0
                if nmb <= size:
                    os.remove(fp)
                    print '{}: {} is less than {} -- removing'.format(fp, nmb, size)


if __name__ == '__main__':

    # Set up commmand line arguments
    parser = argparse.ArgumentParser(description='Download and process data')
    parser.add_argument('--get_all_msd_songs_info', dest='get_all_msd_songs_info', action='store_true')
    parser.add_argument('--match_msd_with_spotify', dest='match_msd_with_spotify', action='store_true')
    # parser.add_argument('--clean_msd_spotify_matched', dest='clean_msd_spotify_matched', action='store_true')
    parser.add_argument('--lookup_spotify_audio_features', dest='lookup_spotify_audio_features', action='store_true')
    parser.add_argument('--clean_spotify_audio_features', dest='clean_spotify_audio_features', action='store_true')
    parser.add_argument('--put_spotify_af_in_db', dest='put_spotify_af_in_db', action='store_true')
    parser.add_argument('--plot_af_stats', dest='plot_af_stats', action='store_true')
    parser.add_argument('--dl_previews_from_spotify', dest='dl_previews_from_spotify', action='store_true')
    parser.add_argument('--clean_dld_previews', dest='clean_dld_previews', action='store_true')
    parser.add_argument('--write_spotify_to_tfrecords', dest='write_spotify_to_tfrecords', action='store_true')
    parser.add_argument('--modify_tfrecords', dest='modify_tfrecords', action='store_true')
    parser.add_argument('--precompute_numpts_and_meanstd_from_tfrecords', dest='precompute_numpts_and_meanstd_from_tfrecords',
                        action='store_true')

    parser.add_argument('--extract_audio_from_vids', dest='extract_audio_from_vids', action='store_true')
    parser.add_argument('--remove_bad_mp3s', dest='remove_bad_mp3s', action='store_true')
    parser.add_argument('--vids_dirpath', dest='vids_dirpath', default=None, help='e.g. data/videos/films')
    parser.add_argument('--bad_mp3_size', dest='bad_mp3_size', type=float, default=0.0, help='upper limit in MB')

    cmdline = parser.parse_args()

    if cmdline.get_all_msd_songs_info:
        pprint(get_all_msd_songs_info())
    elif cmdline.match_msd_with_spotify:
        match_msd_with_spotify()
    # elif cmdline.clean_msd_spotify_matched:
    #     clean_msd_spotify_matched()
    elif cmdline.lookup_spotify_audio_features:
        lookup_spotify_audio_features()
    elif cmdline.clean_spotify_audio_features:
        clean_spotify_audio_features()
    elif cmdline.put_spotify_af_in_db:
        put_spotify_af_in_db()
    elif cmdline.plot_af_stats:
        plot_af_stats()
    elif cmdline.dl_previews_from_spotify:
        dl_previews_from_spotify()
    elif cmdline.clean_dld_previews:
        clean_dld_previews()
    elif cmdline.write_spotify_to_tfrecords:
        write_spotify_to_tfrecords()
    elif cmdline.precompute_numpts_and_meanstd_from_tfrecords:
        precompute_numpts_and_meanstd_from_tfrecords()
    elif cmdline.modify_tfrecords:
        modify_tfrecords()
    elif cmdline.extract_audio_from_vids:
        extract_audio_from_vids(cmdline.vids_dirpath)
    elif cmdline.remove_bad_mp3s:
        remove_bad_mp3s(cmdline.vids_dirpath, cmdline.bad_mp3_size)