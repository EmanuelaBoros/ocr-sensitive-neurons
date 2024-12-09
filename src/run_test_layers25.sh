#!/bin/bash

for num_neurons in {100..10000..100}
do
    for decrease in 0.1 0.5 0.9
    do
        CUDA_VISIBLE_DEVICES=1 python run_ner.py --layer 25 --decrease $decrease --max_length 128 --batch_size 16 --epochs 1 --layer_level Middle_Layers --num_neurons $num_neurons --test
    done
done


