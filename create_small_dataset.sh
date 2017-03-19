#!/bin/bash

SIZE=2000
mkdir -p small_datasets/mscoco_inpainting/inpainting/train2014/
ls -1 datasets/mscoco_inpainting/inpainting/train2014/ | head -n $SIZE | xargs -n1 -I{} cp datasets/mscoco_inpainting/inpainting/train2014/'{}' small_datasets/mscoco_inpainting/inpainting/train2014/
