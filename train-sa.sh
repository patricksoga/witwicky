#!/bin/bash
#$ -q gpu@@nlp-gpu
#$ -l gpu_card=1

touch nmt/DEBUG.log
fsync -d 10 nmt/DEBUG.log &

SL=en
TL=vi
PE=lape_spectral_attn
TASK=${SL}2${TL}_${PE}
conda activate gnnenv
python3 -m nmt --proto $TASK
