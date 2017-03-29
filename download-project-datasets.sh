#!/bin/bash

set -o verbose on

user_dir=/Tmp/$USER
data_dir="$userdir"/mscoco
mkdir -p "$data_dir"
pushd "$user_dir"
#wget http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/TO_READ.txt
#wget http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/examples.py
wget http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/inpainting.tar.bz2
tar xf inpainting.tar.bz2
mv inpainting/* "$data_dir"
rm -rf inpainting*
popd

ln -sf "$data_dir"
