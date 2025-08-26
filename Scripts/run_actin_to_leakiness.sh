#!/bin/bash -x

dd="./data/permeability/"
name="actin_junction_mix_0.1_kernel_5_nlayers_4_ds_100_lrstep_100_ws_200_fl_0_1"
x_prefix="pred_junction"
#x_prefix="actin"
c=0 # Which gpu to use
t=1 # Train

if [ "$#" -ge 1 ];
    then c=$1
fi
if [ "$#" -eq 2 ]; then
	t=0
fi

echo "cuda" $c "train" $t

# Store all predicted junctions in a new data/pred folder.
    if [ $x_prefix == "pred_junction" ]; then
    	python3 CODE/predict.py -x actin -y junction --model_name=$name --device $c --data_path=$dd
    fi

if [ $t -eq 1 ]; then	    
    # Train outline prediction from junction prediction
	python3 CODE/main.py --train -x $x_prefix -y leakiness -e 4 --batch_size=32 --window_size=200 --lr=.000001 --n_window 200 --lr_step=20 --ga 1. --zero_weight=0.4 --leak_thresh=0.1 -a 1 -k 5 -p 2 --n_layers 4 --save 1 --data_path $dd --cuda cuda:$c --pname=$name --bdy=10 --mrg=40

name_leak="$(tail -1 Output/log_leak_$c.txt)" 
echo $name_leak

fi

name_leak="$(tail -1 Output/log_leak_$c.txt)" 
echo $name_leak

 # Store predicted outlines in data/pred folder
python3 CODE/predict.py -x $x_prefix -y leakiness  --zero_thresh=.8  --pred_folder_name=$name --model_name=$name_leak --device $c --data_path=$dd