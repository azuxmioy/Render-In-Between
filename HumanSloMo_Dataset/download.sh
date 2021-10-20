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
cd $VIDEOPATH

gdown https://drive.google.com/uc?id=1OlqoZumoeWyWmoGGrrf3AyHU7Rr1O55P
gdown https://drive.google.com/uc?id=1Fi0U27qA1RS2T5kCI0E4eg3nVdmTKNkj

tar xvf subject1.tar
tar xvf subject2.tar

mv subject1 00_Dance
mv subject2 01_Dance

echo "Finished!!!"