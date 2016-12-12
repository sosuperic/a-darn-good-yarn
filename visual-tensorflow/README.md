# Visual emotion analysis
Includes analysis of positive-negative sentiment and emotions on images and videos.

## symlinks
- List of symlinks set up
    * Each dataset in `data/`
    * `core` in each task dir in `tasks/`
        * These are currently also set up in `bash/setup_symlinks.sh` (and should be added for each subsequent task). They are not done for the datasets because the location of the datasets is up to the user and system.
- To set up a symlink:
    * Go to directory where symlink should be placed, and run `ln -s <actual directory> <name of directory>`
        * Example: in `tasks/image-sent/`, run `ln -s ../../core core`