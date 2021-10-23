#!/bin/bash

INPUTPATH=$1
OUTPUTPATH=$2

if ! [ -d $OUTPUTPATH ]; then
    mkdir $OUTPUTPATH
fi

if [ -d $OUTPUTPATH ]; then
    python3 lib/gen_dataset_h5.py -i "$INPUTPATH" -o "$OUTPUTPATH"
else
    echo "Directory does not exists."
    exit 0
fi

echo "Finished!!!"