#!/bin/bash

SIZE=2000

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

pushd "$user_dir" &>/dev/null
mkdir -p mscoco_small/train2014
cp mscoco/*.pkl mscoco_small/
ls -1 mscoco/train2014/ | head -n $SIZE | xargs -n1 -I{} cp mscoco/train2014/'{}' mscoco_small/train2014/
popd &>/dev/null

ln -sf "${user_dir}/mscoco_small" mscoco
