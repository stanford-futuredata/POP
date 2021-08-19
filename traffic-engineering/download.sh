#! /usr/bin/env bash

set -e
set -x

export paths_file_id=1kmWab5GUHKLwTIbefKVThsMSdwh953bh
export paths_filename=paths.zip

export traffic_matrices_file_id=1cX9pzQmUXjArFU0q3SbwRN6WT5tb9B4I
export traffic_matrices_filename=traffic-matrices.zip

# Borrowed from https://www.matthuisman.nz/2019/01/download-google-drive-files-wget-curl.html
curl -L -c cookies.txt 'https://docs.google.com/uc?export=download&id='$paths_file_id \
       | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
curl -L -b cookies.txt -o $paths_filename \
       'https://docs.google.com/uc?export=download&id='$paths_file_id'&confirm='$(<confirm.txt)
rm -f confirm.txt cookies.txt
unzip paths.zip && rm paths.zip

curl -L -c cookies.txt 'https://docs.google.com/uc?export=download&id='$traffic_matrices_file_id \
       | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
curl -L -b cookies.txt -o $traffic_matrices_filename \
       'https://docs.google.com/uc?export=download&id='$traffic_matrices_file_id'&confirm='$(<confirm.txt)
rm -f confirm.txt cookies.txt
unzip traffic-matrices.zip && rm traffic-matrices.zip
