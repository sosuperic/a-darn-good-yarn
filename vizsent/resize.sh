#!/bin/bash
#regex="[abcdef]"
cd Sentibank/Flickr/tmp/
for d in */ ; do
    #echo $d
    # if [[ $d  =~ ^[ghijklmnopqrstuvwxyz].* ]];
    # then
        echo "$d";
        (cd $d && parallel convert {} -resize 256x256^ {} ::: *.jpg)
    # fi
    #(cd $d && find . -name '*.jpg' | xargs -I {} convert {} -resize "256^>" {})
done
