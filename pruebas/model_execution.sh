#!/bin/bash

echo "Starting fine tuning for GPT2 on medical text data"
echo "Unzipping data..."
# unzip data
unzip data.zip

echo ""
echo ""
echo "Generating folder structure and installing dependencies"
# generate folder structure
mkdir output

# install dependencies
pip3 install git+https://github.com/huggingface/transformers --quiet
pip3 install datasets --quiet
pip3 install torch --quiet
pip3 install numpy --quiet


echo ""
echo ""
echo "Prepared to run the model"
# run model on data

python3 src/run_clm.py \
    --model_name_or_path gpt2 \
    --train_file tagged_files/train_tagged_prueba.txt\
    --per_device_train_batch_size 32 \
    --do_train \
    --output_dir output

echo ""
echo ""
echo "Training finished, zipping it all back"
# zip all back
zip results.zip output/