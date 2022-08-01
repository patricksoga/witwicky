#!/bin/bash
#$ -q *@@nlp-gpu
#$ -l gpu_card=1

SL=en
TL=vi
TASK=${SL}2${TL}
module load pytorch/1.1.0
ls nmt/saved_models/$TASK/$TASK.pth
ls nmt/data/$TASK/$1.$SL
python3 -m nmt --mode translate --proto $TASK --model-file nmt/saved_models/$TASK/$TASK.pth --input-file nmt/data/$TASK/$1.$SL
