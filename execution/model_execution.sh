#!/bin/bash

# unzip data
unzip data.zip

# generate folder structure
mkdir output

# install dependencies
pip3 install git+https://github.com/huggingface/transformers
pip3 install datasets

# run model on data

python3 run_clm.py \
    --model_name_or_path gpt2 \
    --train_file ../tagged_files/train_tagged_prueba.txt\
    --per_device_train_batch_size 1 \
    --do_train \
    --output_dir output


# zip all back
zip . results.zip