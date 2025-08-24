#!/bin/bash -x

$dd='data/'

c=$1
name=$2
name1=$3

# Test outline prediction directly from actin using the two models.
python3 CODE/main.py --test --direct_test=1 --model_name=$name --model_name_o=$name1 --cuda cuda:$c --data_path=$dd

