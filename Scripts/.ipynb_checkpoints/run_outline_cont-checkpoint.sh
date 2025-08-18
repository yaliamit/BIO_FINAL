#!/bin/bash -x

c=0
r=0

if [ "$#" -ge 1 ];
    then c=$1
fi
if [ "$#" -eq 2 ] ;
    then r=$2
fi
echo "cuda" $c "reduced" $r

name=actin_junction_mix_0.1_kernel_5_margin_0_nlayers_4_ds_0_reduced_0_SA_1_lrstep_50_2
python3 main.py --train -x actin -y junction -e 200 --lr=.0000001 --cont 1 --dice_steps=0 --lr_step=100 --ga=.5 --save 1 --n_layers=4 --cuda cuda:$c --SA 1 --model_name $name

name="$(tail -1 log2.txt)" 
echo $name

python3 predict.py -x actin -y junction  -p 0 -n 1 --name=$name --device 2

python3 main.py --train -x pred_junction -y outline -e 200 --lr=.00001 --lr_step=100 -a 0.0 -k 3 -p 1 --n_layers 4 --save 1 --reduced $r --out_margin 0 --SA 1 --data_path "./data/pred/"$name --cuda cuda:2


name1="$(tail -1 log2.txt)" 
echo $name1

for i in '_50' '_100' '_150' '';
    do
     python3 main.py --test -x pred_junction -y outline --reduced 0 --model_name=$name1$i --data_path "./data/pred/"$name --cuda cuda:2
    done

name1=pred_junction_outline_mix_0.0_kernel_3_margin_0_nlayers_4_ds_0_reduced_0_SA_1_2
python3 predict.py -x pred_junction -y outline -p 0 -n 1 --name1=$name --name=$name1 --device 2