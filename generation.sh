#!/bin/bash
python3 data_generation.py \
    --output_path ./generation/imseg \
    --desired_size 1024 \
    --workload imseg


python3 data_generation.py \
    --output_path ./generation/dlrm \
    --desired_size 1024 \
    --workload dlrm \
    --data_format bin

python3 data_generation.py \
    --output_path ./generation/bert \
    --desired_size 1024 \
    --workload bert
