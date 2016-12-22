#!/bin/bash

# If checkpoint directory only has params.json, either (a) an error was thrown, or (b) I terminated it early.
# Keeping these around just clutters things

cd tasks
for task_dir in */ ; do
    cd $task_dir/checkpoints
    for ckpt_dir in */ ; do
        if [ -d ${ckpt_dir} ]; then        # will not run if no directories are available
            if [ $(ls $ckpt_dir | wc -l) == 1 ]; then
                if [ $(ls $ckpt_dir) == "params.json" ]; then
                    rm -rf $ckpt_dir
                fi
            fi
        fi
    done
    cd ../..
done