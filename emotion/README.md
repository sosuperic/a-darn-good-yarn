# Visual emotion analysis
Includes analysis of positive-negative sentiment and emotions on images and videos.

## website
- Used to plot shape and adjust window size, see frames at various points of curve
- symlink website/shape/static/videos -> <videos path>
- Run in website/ by:
    * `python run.py` (local)
        * visit: `http://localhost:7898/shape`
    * `gunicorn -w 2 -b 0.0.0.0:7898 run:app` (on Shannon)
        * visit: `http://18.85.54.48:7898/shape`

## symlinks
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