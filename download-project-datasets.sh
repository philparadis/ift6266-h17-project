#!/bin/bash

set -o verbose on

SAVE_DIR=""

if [[ -d "/Tmp" ]]; then
    SAVE_DIR="/Tmp"
elif [[ -d "/tmp" ]]; then
    SAVE_DIR="/tmp"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
    SAVE_DIR="$SCRIPT_DIR"
fi

user_dir="${SAVE_DIR}/${USER}"
data_dir="$user_dir"/mscoco
mkdir -p "$data_dir"
pushd "$user_dir"
#wget http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/TO_READ.txt
#wget http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/examples.py
file=inpainting.tar.bz2
skip_download=0
if [[ -f "$file" ]]; then
    if [[ $(sha1sum "$file" | awk '{print $1}') == "4da23a7b4b7a016de38c7f7281bffa55649ce91c" ]]; then
	echo "Okay, file '$file' already exists and checksum is valid, skipping download."
	skip_download=1
    fi
fi

if [[ "$skip_download" == 0 ]]; then
    wget http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/"$file"
fi

tar xf inpainting.tar.bz2
mv inpainting/* "$data_dir"
rm -r inpainting
popd

ln -sf "$data_dir"
