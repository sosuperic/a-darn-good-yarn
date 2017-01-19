# Prepare the data (parse, convert to tfrecords, download, etc.)

import argparse
from collections import defaultdict, Counter
from fuzzywuzzy import fuzz
import io
import json
import os
import pickle
from pprint import pprint
import re
import sqlite3
import subprocess

from core.utils.MovieReader import MovieReader
from core.utils.CreditsLocator import CreditsLocator


# Videos path
VIDEOS_PATH = 'data/videos'

# CMU Movie Summary path
CMU_PATH = 'data/CMU_movie_summary/MovieSummaries/'

# Videos
VID_EXTS = ['webm', 'mkv', 'flv', 'vob', 'ogv', 'ogg', 'drc', 'gif', 'gifv', 'mng', 'avi', 'mov', 'qt', 'wmv',
                'yuv', 'rm', 'rmvb', 'asf', 'amv', 'mp4', 'm4p', 'm4v', 'mpg', 'mp2', 'mpeg', 'mpe', 'mpv', 'm2v',
                'm4v', 'svi', '3gp', '3g2', 'mxf', 'roq', 'nsv', 'flv', 'f4v', 'f4p', 'f4a', 'f4b']
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
def save_video_frames(vids_dir):
    """
    Loop over subdirs within vids_dir and save frames to subdir/frames/

    Parameters
    ----------
    vids_dir: directory within VIDEOS_PATH that contains sub-directories, each which may contain a movie
    """

    # vid_exts = ['mp4', 'avi']
    mr = MovieReader()

    vids_path = os.path.join(VIDEOS_PATH, vids_dir)
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
                mr.write_frames(movie_path)
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

    }

    result = {}

    conn = sqlite3.connect(VIDEOPATH_DB)
    with conn:
        cur = conn.cursor()
        rows = cur.execute("SELECT title FROM VideoPath WHERE category=='films'")
        for row in rows:
            title = row[0]
            title = title.encode('utf-8')
            m = re.match(r'(.+) \(\d+\)?$', title)
            title = m.group(1)

            if title in manually_matched:
                print title, 'TITLE IN MANUALLY MATCHED'
                match = manually_matched[title]
                if match:
                    result[title] = movie2metadata[match]
                    continue
            else:
                matched = sorted([(fuzz.ratio(title, movie_name), movie_name) for movie_name in movies])
                matched = [t for s, t in matched[::-1][:10]]        # top 10
                match = matched[0]
                result[title] = movie2metadata[match]

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
    parser.add_argument('--convert_avis_to_mp4s', dest='convert_avis_to_mp4s', action='store_true')
    parser.add_argument('--save_credits_index', dest='save_credits_index', action='store_true')
    parser.add_argument('--save_credits_index_overwrite', dest='save_credits_index_overwrite', default=False,
                        action='store_true', help='overwrite credits_index.txt files')
    parser.add_argument('--create_videopath_db', dest='create_videopath_db', action='store_true')
    parser.add_argument('--match_film_metadata', dest='match_film_metadata', action='store_true')
    parser.add_argument('--get_shorts_metadata', dest='get_shorts_metadata', action='store_true')
    parser.add_argument('--create_videometadata_db', dest='create_videometadata_db', action='store_true')
    parser.add_argument('--vids_dir', dest='vids_dir', default=None,
                        help='folder that contains dirs (one movie each), e.g. films/MovieQA_full_movies')

    cmdline = parser.parse_args()

    if cmdline.save_video_frames:
        save_video_frames(cmdline.vids_dir)
    elif cmdline.convert_avis_to_mp4s:
        convert_avis_to_mp4s(cmdline.vids_dir)
    elif cmdline.save_credits_index:
        save_credits_index(cmdline.vids_dir, overwrite_files=cmdline.save_credits_index_overwrite)
    elif cmdline.create_videopath_db:
        create_videopath_db()
    elif cmdline.match_film_metadata:
        pprint(match_film_metadata())
    elif cmdline.get_shorts_metadata:
        pprint(get_shorts_metadata())
    elif cmdline.create_videometadata_db:
        create_videometadata_db()