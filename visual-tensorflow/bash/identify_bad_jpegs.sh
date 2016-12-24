#!/bin/bash

# Note: error output from identify doesn't always contain file image...
# Example: identify -format "%f" colorful_building/6117947847_109c06bbb8.jpg
# Running `file 6117947847_109c06bbb8.jpg` actually returns `HTML document, ASCII text, with CRLF line terminators`
# Output is:
#    identify.im6: delegate failed `"html2ps" -U -o "%o" "%i"' @ error/delegate.c/InvokeDelegate/1065.
#    identify.im6: unable to open image `/tmp/magick-17BPcJOP': No such file or directory @ error/blob.c/OpenBlob/2641.
#    identify.im6: unable to open file `/tmp/magick-17BPcJOP': No such file or directory @ error/constitute.c/ReadImage/583.
# Therefore, use the difference between files and ok_jpgs files as bad jpegs

if [[ $# -eq 0 ]] ; then
    echo 'Pass in path to directory with subdirectories, each subdir containing images'
    exit 0
fi

# Make folder to contain outputs
parentdir="$(dirname "$1")"
mkdir -p $parentdir/jpg_check
rm -f $parentdir/error_jpgs.txt
touch $parentdir/error_jpgs.txt

cd $1
for d in */ ; do
    echo "$d"
    cd $d
    find -name '*.jpg' -exec identify -format "%f" {} \; 1>ok_jpgs.txt 2>error_jpgs.txt

    # Add it to main error file
    if (( $(cat error_jpgs.txt | wc -l) > 0 )); then
        echo "$d" >> ../../error_jpgs.txt
        cat error_jpgs.txt >> ../../error_jpgs.txt
    fi

    # Move files to jpg_check folder
    mkdir -p ../../jpg_check/$d
    mv ok_jpgs.txt ../../jpg_check/$d/
    mv error_jpgs.txt ../../jpg_check/$d/

    cd ..
done
