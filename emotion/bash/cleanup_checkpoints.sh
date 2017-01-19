#!/bin/bash

# If checkpoint directory doesn't have checkpoint, either (a) an error was thrown, or (b) I terminated it early.
# Keeping these around just clutters things

cd tasks
for task_dir in */ ; do
    cd $task_dir/checkpoints
    for ckpt_dir in */ ; do
        if [ -d ${ckpt_dir} ]; then        # will not run if no directories are available
            if [ ! -e "$ckpt_dir/checkpoint" ]; then
                rm -rf $ckpt_dir
            fi
        fi
    done
    cd ../..
done