# Prepare the data (parse, convert to tfrecords, download, etc.)

import argparse
import cPickle as pickle
from multiprocessing.dummy import Pool as ThreadPool
from natsort import natsorted
import os
from pprint import pprint
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import sqlite3
import string
import urllib

MSD_FILES_PATH = 'data/msd/full/AdditionalFiles/'
MSD_TRACKS_DB = 'data/msd/full/AdditionalFiles/track_metadata.db'
MSD_SPOTIFY_MATCH_RESULTS  = 'data/msd/full/AdditionalFiles/msd_spotify_match_results_{}.pkl'

SPOTIFY_PATH = 'data/spotify/'
SPOTIFY_PREVIEWS_PATH = 'data/spotify/previews/'
MSD_SPOTIFY_MATCH_DB = 'data/spotify/MsdSpotifyMatch.db'
SPOTIFY_AUDIO_FEATURES = 'data/spotify/spotify_audio_features.pkl'

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
    msmr_out_fp = os.path.join(MSD_SPOTIFY_MATCH_DB, MSD_SPOTIFY_MATCH_RESULTS.format(cur_msmr_idx))
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
    i = 0                                   # number looked up this run, just used for printing
    for spotify_track_id in all_spotify_track_ids:
        if spotify_track_id in lookedup_so_far:
            # print 'Track already looked up, skipping'
            continue
        else:
            cur_batch_spotify_track_ids.append(spotify_track_id)

        if len(cur_batch_spotify_track_ids) % 50 == 0:
            # Lookup and save features
            tracks_features = sp.audio_features(cur_batch_spotify_track_ids)
            for j, track_features in enumerate(tracks_features):
                lookedup_so_far[cur_batch_spotify_track_ids[j]] = track_features
            with open(SPOTIFY_AUDIO_FEATURES, 'wb') as f:
                pickle.dump(lookedup_so_far, f, protocol=2)

            # Reset
            cur_batch_spotify_track_ids = []

            # Print a count
            i += 50
            print '=' * 100
            print 'cur_run: {}, looked_up_so_far: {}, num_match: {}'.format(
                i, len(lookedup_so_far), len(all_spotify_track_ids))
            print '=' * 100

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
    songs_per_folder = 100   # 26 * 26 * 26 * 75 = 1757600
    alph = string.ascii_uppercase

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
    dir_indices = open(os.path.join(SPOTIFY_PATH, 'previews_dir_indices.txt'), 'rb').readlines()[0].split(',')
    dir_indices = [int(s) for s in dir_indices]
    cur_alph_indices = dir_indices[0:3]
    i = dir_indices[3]
    print cur_alph_indices, i

    # Check that indices are correct
    num_files_idx = (songs_per_folder * cur_alph_indices[2]) + \
            (songs_per_folder * (cur_alph_indices[1] * 26)) + \
            (songs_per_folder * (cur_alph_indices[0] * 26 * 26)) + \
            i   # i is supposed to be index for next file to save, so if A/Z/B/ has one file, i = 1
    if num_files_idx != num_already_downloaded:
        print 'Indices incorrect: {} vs. {}'.format(num_files_idx, num_already_downloaded)
        return


    # Get spotify_track_ids that have been matched with msd
    conn = sqlite3.connect(MSD_SPOTIFY_MATCH_DB)
    cur = conn.cursor()
    spotify_track_ids_and_preview_urls = []
    rows = cur.execute("SELECT spotify_track_id, spotify_preview_url FROM MsdSpotifyMatch "
                       "WHERE (spotify_track_id is not NULL) AND (spotify_preview_url is not NULL)")
    for row in rows:
        spotify_track_ids_and_preview_urls.append([row[0], row[1]])

    print 'Getting urls and fps'
    urls = []
    fps = []
    for spotify_track_id, preview_url in spotify_track_ids_and_preview_urls:
        # Skip if preview mp3 already downloaded
        if spotify_track_id in already_downloaded:
            continue

        urls.append(preview_url)

        # Get fiilepath
        dir = alph[cur_alph_indices[0]] + '/' + alph[cur_alph_indices[1]] + '/' + alph[cur_alph_indices[2]]
        dirpath = os.path.join(SPOTIFY_PREVIEWS_PATH, dir)
        if not os.path.exists(dirpath):
            os.system('mkdir -p {}'.format(dirpath))
        fp = os.path.join(dirpath, '{}.mp3'.format(spotify_track_id))
        fps.append(fp)
        print fp

        # Update fp indices
        if (i != 0) and (i % songs_per_folder == 0):
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

if __name__ == '__main__':

    # Set up commmand line arguments
    parser = argparse.ArgumentParser(description='Download and process data')
    parser.add_argument('--get_all_msd_songs_info', dest='get_all_msd_songs_info', action='store_true')
    parser.add_argument('--match_msd_with_spotify', dest='match_msd_with_spotify', action='store_true')
    # parser.add_argument('--clean_msd_spotify_matched', dest='clean_msd_spotify_matched', action='store_true')
    parser.add_argument('--lookup_spotify_audio_features', dest='lookup_spotify_audio_features', action='store_true')
    parser.add_argument('--clean_spotify_audio_features', dest='clean_spotify_audio_features', action='store_true')
    parser.add_argument('--dl_previews_from_spotify', dest='dl_previews_from_spotify', action='store_true')
    parser.add_argument('--spotify_key', dest='spotify_key', default=None)

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
    elif cmdline.dl_previews_from_spotify:
        dl_previews_from_spotify()