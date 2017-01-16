# Process .srt subtitle files
# Iterate over folders in MOVIES_PATH and find .srt file

import os
import pysrt
import pickle
import csv


MOVIES_PATH = '/Users/eric/Movies'

def process_subs(path):
	"""
	Save text, start_min, start_sec, end_min, end_sec to pkl and csv
	"""
	print('Processing: ' + path)
	subs = pysrt.open(path)

	subs_pkl = []

	movie_dir = os.path.dirname(path)
	with open(os.path.join(movie_dir, 'subs.csv'), 'wb') as csvfile:
		writer = csv.writer(csvfile,delimiter='|')
		for s in subs:
			text = s.text.encode('utf-8')	# TODO: errors with invalid continuation byte / invalid start byte
			text = text.replace("\n", " ")
			sh, sm, ss = s.start.hours, s.start.minutes, s.start.seconds
			eh, em, es = s.end.hours, s.end.minutes, s.end.seconds
			writer.writerow([text, sh, sm, ss, eh, em, es])
			subs_pkl.append([text, sh, sm, ss, eh, em, es])

	f = open(os.path.join(movie_dir, 'subs.pkl'), 'wb')
	pickle.dump(subs_pkl, f, protocol=2)
	f.close()


def process_all_subs():
	dirs = [d for d in os.listdir(MOVIES_PATH) if os.path.isdir(os.path.join(MOVIES_PATH, d))]
	for d in dirs:
		files = os.listdir(os.path.join(MOVIES_PATH, d))
		for f in files:
			if f.endswith('.srt'):
				path = os.path.join(MOVIES_PATH, d, f)
				try: 
					process_subs(path)
				except UnicodeDecodeError as e:
					print(e)

if __name__ == '__main__':
	# process_all_subs()
	process_subs('/Users/eric/Movies/films/Other/Ant Man (2015)/English.srt')