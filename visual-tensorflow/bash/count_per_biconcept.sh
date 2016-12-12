#!/bin/bash

cd data/Sentibank/Flickr/bi_concepts1553
for d in */ ; do
    echo -e "$d\c"
    (cd $d && find . -name '*.jpg' | wc -l)
done
