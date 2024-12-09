#!/bin/bash

for num_neurons in {1000..10000..1000}
do
    for decrease in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        CUDA_VISIBLE_DEVICES=0 python run_ner.py --decrease $decrease --max_length 128 --batch_size 16 --epochs 3 --layer_level First_Layers --num_neurons $num_neurons
    done
done

