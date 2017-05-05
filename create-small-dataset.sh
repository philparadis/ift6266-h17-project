#!/bin/bash

TRAIN_SIZE=5000
TEST_SIZE=500

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
small_target_dir="mscoco_small"

pushd "$user_dir" &>/dev/null
mkdir -p "${small_target_dir}"/train2014
mkdir -p "${small_target_dir}"/test2014
cp "${target_dir}"/*.pkl "${small_target_dir}"/
ls -1 "${target_dir}"/train2014/ | head -n $TRAIN_SIZE | xargs -n1 -I{} cp "${target_dir}"/train2014/'{}' "${small_target_dir}"/train2014/
ls -1 "${target_dir}"/test2014/ | head -n $TEST_SIZE | xargs -n1 -I{} cp "${target_dir}"/test2014/'{}' "${small_target_dir}"/test2014/
popd &>/dev/null

if [[ -s "${target_dir}" ]]; then
    rm "${target_dir}"
fi
ln -sf "${user_dir}/${small_target_dir}" "${target_dir}"
