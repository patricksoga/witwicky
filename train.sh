#!/bin/bash
#$ -q *@@nlp-gpu
#$ -l gpu_card=1

touch nmt/DEBUG.log
fsync -d 10 nmt/DEBUG.log &

SL=en
TL=vi
PE=lape_big_graph
TASK=${SL}2${TL}_${PE}
conda activate gnn
python3 -m nmt --proto $TASK
