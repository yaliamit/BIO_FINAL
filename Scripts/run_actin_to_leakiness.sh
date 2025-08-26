#!/bin/bash -x

dd="./data/permeability/"
name="actin_junction_mix_0.1_kernel_5_nlayers_4_ds_100_lrstep_100_ws_200_fl_0_1"

c=0 # Which gpu to use
t=1 # Train

if [ "$#" -ge 1 ];
    then c=$1
fi
if [ "$#" -eq 2 ]; then
	t=0
fi

echo "cuda" $c "train" $t

if [ $t -eq 1 ]; then	

    # Store all predicted junctions in a new data/pred folder.
	#python3 CODE/predict.py -x actin -y junction -p 0 -n 1 --name=$name --device $c --data_path=$dd

    # Train outline prediction from junction prediction
	python3 CODE/main.py --train -x pred_junction -y leakiness -e 200 --batch_size=32 --window_size=200 --lr=.000001 --n_window 200 --lr_step=20 --ga 1. --zero_weight=0.4 --leak_thresh=0.1 -a 1 -k 5 -p 2 --n_layers 4 --save 1 --data_path $dd --cuda cuda:$c --pname=$name --bdy=10 --mrg=40

name_leak="$(tail -1 Output/log$c.txt)" 
echo $name_leak

fi

name_leak="$(tail -1 Output/log_leak_$c.txt)" 
echo $name_leak

 # Store predicted outlines in data/pred folder
python3 CODE/predict.py -x pred_junction -y leakiness  --zero_thresh=.8  --pred_folder_name=$name --model_name=$name_leak --device $c --data_path=$dd