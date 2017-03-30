#!/bin/bash

#set -o verbose on

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
    if [[ "$digest" == "592931a25a997aa0c2f012a3a25aaa2eb1201df4" ]]; then
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
fi

popd &>/dev/null

echo "Creating a symlink $(pwd)/${target_dir} --> ${user_dir}/${target_dir}"
ln -sf "${user_dir}/${target_dir}"
