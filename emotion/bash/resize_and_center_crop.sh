#!/bin/bash

cd $1
for d in */ ; do
    echo "$d";
    (cd $d && parallel convert {} -resize 256x256^ {} ::: *.jpg && cd ..)
    (cd $d && parallel convert {} -gravity Center -crop 256x256+0+0 +repage {} ::: *.jpg)
done
