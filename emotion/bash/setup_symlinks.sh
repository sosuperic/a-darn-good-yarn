#!/bin/bash

# Have to do it within the folder for folders for some reason, hence the cd
# -L checks if file exists and is a symlink
# -d checks if file is a directory

# Linking folders to tasks
if ! [[ -L tasks/image-sent/core && -d tasks/image-sent/core ]]
then
    cd tasks/image-sent/ && ln -s ../../core core
    echo "Set symlink from tasks/image-sent/core to core"
fi