#!/bin/bash -x

dd="./data"

c=0 # Which gpu to use
r=0 # Reduced outline classes to 3 (=1) or all 4 (=0)
t=1 # Train
if [ "$#" -ge 1 ];
    then c=$1
fi
if [ "$#" -ge 2 ];
    then r=$2
fi
if [ "$#" -eq 3 ];
    then t=$3
fi



if [ $t -eq 1 ]; then
    python3 CODE/main.py --train -x junction -y outline -e 200 --lr=.00001 --lr_step=100 --kernel_size 5 --padding 2 -a 0.0 --save 1 --reduced $r --margin 0 --data_path $dd --ga 1. --cuda cuda:$c --bdy=10 --mrg=40
    name_outline="$(tail -1 Output/log$c.txt)"    
else
    name_outline=$t
fi
echo $name_outline

# Run the model on test data and report classification rates.
for i in '';
  do
   python3 CODE/main.py --test -x junction -y outline -a 0.0 --reduced=$r --model_name=$name_outline$i --data_path $dd --cuda cuda:$c
  done