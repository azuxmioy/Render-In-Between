#!/bin/bash

VIDEOPATH=$1
CSV_FILE=$2
TEMP_FILE="tmp.sh"

if ! [ -d $VIDEOPATH ]; then
    mkdir $VIDEOPATH
fi

if [ -d $VIDEOPATH ]; then
    python3 lib/download.py -o $VIDEOPATH -c $CSV_FILE -t $TEMP_FILE
    if [ -f $TEMP_FILE ]; then
        bash $TEMP_FILE
    else
        echo "File $TEMP_FILE does not exists."
    fi
else
    echo "Directory does not exists."
    exit 0
fi

rm $TEMP_FILE
echo "Finished!!!"