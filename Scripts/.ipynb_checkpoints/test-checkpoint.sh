#!/bin/bash -x

c=$1
name=$2
name1=$3
#name="actin_junction_mix_0.1_kernel_5_margin_0_nlayers_4_ds_100_reduced_0_0"
#name1="pred_junction_outline_mix_0.0_kernel_5_margin_0_nlayers_4_ds_0_reduced_1_0"

python3 CODE/main.py --test --direct_test=1 --model_name=$name --model_name_o=$name1 --cuda cuda:$c

