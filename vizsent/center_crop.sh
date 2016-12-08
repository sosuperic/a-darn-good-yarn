#!/bin/bash
#regex="[abcdef]"
cd Sentibank/Flickr/tmp/
for d in */ ; do

    # echo $d
    # if [[ $d  =~ ^[abcdefghijklmnopqrstuvwxyz].* ]];
    # then
        echo "$d";
        (cd $d && parallel convert {} -gravity Center -crop 256x256+0+0 +repage {} ::: *.jpg)
    # fi
done
