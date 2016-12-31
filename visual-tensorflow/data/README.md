- Note on downloading videos from Youtube/Vimeo:
    * Downloaded 1441 videos from Vimeo 'Short of the Week' playlist using the extremely robust youtube-dl 
    * Link: https://github.com/rg3/youtube-dl
    * Installation:
        * `sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl`
        * `sudo chmod a+rx /usr/local/bin/youtube-dl`
    * Command used in `/media/lsm/dude/storytelling_datasets/videos/shorts/`
        * `youtube-dl -o 'shortoftheweek/%(title)s_%(id)s/%(title)s_%(id)s.%(ext)s' https://vimeo.com/channels/shortoftheweek --write-sub --write-description --write-info-json --write-annotations  --write-thumbnail --ignore-errors -f 'mp4+bestvideo[height<=480][height>=256]+bestaudio/best[height<=480][height>=256]' --recode-video mp4`