#!/bin/bash

set -o verbose on

if [[ ! $(basename $(pwd)) == "datasets" ]]; then
    echo "ERROR: Please run this script from within the 'datasets' directory."
    exit -1
fi

dirname=/Tmp/mscoco_inpainting
mkdir -p "$dirname"
pushd "$dirname"
wget http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/TO_READ.txt
wget http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/examples.py
wget http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/inpainting.tar.bz2
tar xf inpainting.tar.bz2
#rm inpainting.tar.bz2
popd

ln -sf "$dirname"
