#!/bin/bash -x

c=0
r=0
t=1
if [ "$#" -ge 1 ];
    then c=$1
fi
if [ "$#" -eq 2 ];
    then r=$2
fi
if [ "$#" -eq 3 ]; then
	t=0
fi
echo "cuda" $c "reduced" $r "train" $t

if [ $t -eq 1 ]; then	
	python3 CODE/main.py --train -x actin -y junction -e 400 --lr .00001 --dice_steps 100 --lr_step 100 --ga 1. --save 1 --n_layers=4 --cuda cuda:$c
	name="$(tail -1 Output/log$c.txt)" 
	echo $name
	python3 CODE/predict.py -x actin -y junction -p 0 -n 1 --name=$name --device $c

	python3 CODE/main.py --train -x pred_junction -y outline -e 400 --lr=.00001 --lr_step=100 --ga=1. --kernel_size 5 --padding 2 -a 0.0 --save 1 --reduced $r --margin 0 --data_path "./data/pred/"$name --cuda cuda:$c --bdy=10 --mrg=40

	name_outline="$(tail -1 Output/log$c.txt)" 
	echo $name_outline
fi

for i in '_50' '_100' '_150' '';
  do
   python3 CODE/main.py --test -x pred_junction -y outline -a 0.0 --reduced $r --model_name=$name_outline$i --data_path "./data/pred/"$name --cuda cuda:$c
  done
  
python3 CODE/predict.py -x pred_junction -y outline  -p 0 -n 1 --name1=$name --name=$name_outline --device $c
