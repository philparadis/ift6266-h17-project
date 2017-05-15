#!/bin/bash

#set -o verbose on

SAVE_DIR=""
TESTING_DATASET_SIZE=1050

if [[ -d "/Tmp" ]]; then
    SAVE_DIR="/Tmp"
elif [[ -d "/tmp" ]]; then
    SAVE_DIR="/tmp"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
    SAVE_DIR="$SCRIPT_DIR"
fi

user_dir="${SAVE_DIR}/${USER}"
target_dir="mscoco"

mkdir -p "$user_dir"

pushd "$user_dir" &>/dev/null
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

skip_extract=0
if [[ -d "${target_dir}" ]]; then
    echo "Found directory '${target_dir}' already exists. Verifying integrity of data..."
    digest=$(find "${target_dir}" -type f \( -iname '*.pkl' -o -iname '*.jpg' \) -print0 | sort -z | xargs -0 -n5000 cat | openssl dgst -sha1 | sed 's/(stdin)= //')
    if [[ "$digest" == "da39a3ee5e6b4b0d3255bfef95601890afd80709" ]]; then
	echo "Okay, directory is valid! We can skip extracting the archive."
	skip_extract=1
    else
	echo "Directory's content does not validate. Removing directory."
	rm -rf "${target_dir}"
    fi
fi

if [[ "$skip_extract" == 0 ]]; then
    echo "Extracting archive..."
    tar xf inpainting.tar.bz2

    mv inpainting "$target_dir"

    echo "Creating directory 'test2014' as a testing dataset made up of the first ${TESTING_DATASET_SIZE} images from subdirectory 'val2014'"
    pushd "$target_dir" &>/dev/null
    mkdir -p test2014
    ls -1 val2014/ | head -n ${TESTING_DATASET_SIZE} | sed 's/COCO_val2014_//' | xargs -n1 -I{} cp val2014/COCO_val2014_'{}' test2014/COCO_test2014_'{}'
    popd &>/dev/null
fi

popd &>/dev/null

echo "Creating a symlink $(pwd)/${target_dir} --> ${user_dir}/${target_dir}"
if [[ -s "${target_dir}" ]]; then
    rm "${target_dir}"
fi
ln -sf "${user_dir}/${target_dir}"
