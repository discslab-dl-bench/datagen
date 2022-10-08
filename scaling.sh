#!/bin/bash
python3 data_scaling.py \
    --input_path /raid/data/unet/original_dataset/Original_dataset_500GB \
    --output_path ./scaling/imseg \
    --desired_size 1024 \
    --workload imseg

python3 data_scaling.py \
    --input_path /raid/data/bert/preproc_data \
    --output_path ./scaling/bert \
    --desired_size 1024 \
    --workload bert

python3 data_scaling.py \
    --input_path /raid/data/bert/preproc_data \
    --output_path ./scaling/bert \
    --desired_size 1024 \
    --workload bert