#!/bin/bash -x

c=0
cn=0
if [ "$#" -ge 1 ]; then
    c=$1
fi
if [ "$#" -ge 2 ]; then
    m=$2
    cn=1
fi

python3 CODE/main.py --train -x junction -y leakiness -e 200 --batch_size=32 --window_size=200 --lr=.000001 --n_window 200 --lr_step=20 --ga 1. --zero_weight=0.4 --leak_thresh=0.1 -a 1 -k 5 -p 2 --n_layers 4 --save 1 --data_path "./data/permeability/" --cont=$cn --model_name=$m --cuda cuda:$c

