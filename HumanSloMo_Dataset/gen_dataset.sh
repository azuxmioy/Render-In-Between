#!/bin/bash

DATASETPATH=$1

if ! [ -d $DATASETPATH ]; then
    mkdir $DATASETPATH
fi

if [ -d $DATASETPATH ]; then
    python3 lib/gen_dataset.py -i "videos" -j "metadata/test_input.json" -o "$DATASETPATH/test/inputs"
    python3 lib/gen_dataset.py -i "videos" -j "metadata/test_list.json" -o "$DATASETPATH/test/gt"
    python3 lib/gen_dataset.py -i "videos" -j "metadata/train_list.json" -o "$DATASETPATH/train/frames"
else
    echo "Directory does not exists."
    exit 0
fi

echo "Finished!!!"