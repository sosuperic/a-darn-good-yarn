# Multi-modal analysis of emotional content in videos

Producing emotional curves based on visual, audio, and text. Labels include positive-negative sentiment, energy, emotions, and emotional concepts. These emotional curves are then clustered using dynamic time warping distance as a distance metric.

## Current test corpora
- ~500 films
- ~1400 Vimeo shorts

## GUI
- Used to plot emotional curves and view clusterings of curves
- symlink `website/shape/static/videos` -> `<videos path>`
- Run in website/ by:
    * On local: `python run.py`
        * visit: `http://localhost:7898/shape`
    * On shannon: `gunicorn -w 2 -b 0.0.0.0:7898 run:app`
        * visit: `http://18.85.54.48:7898/shape`

## Running

#### Tasks
Each task in `tasks/` allows one to train, test, and predict with a number of models for a given modality.

#### Setting up symlinks
- List of symlinks set up
    * Each dataset in `data/`
        * `MVSO`, `Sentibank`, `emolex`, `videos`, `you_imemo`
    * `core` in each task dir in `tasks/`
        * These are currently also set up in `bash/setup_symlinks.sh` (and should be added for each subsequent task). They are not done for the datasets because the location of the datasets is up to the user and system.
    * `core` in `website/shape/`
    * `videos` in `website/shape/static`
    * `outputs` in `website/shape/`
- To set up a symlink:
    * Go to directory where symlink should be placed, and run `ln -s <actual directory> <name of directory>`
        * Example: in `tasks/image-sent/`, run `ln -s ../../core core`