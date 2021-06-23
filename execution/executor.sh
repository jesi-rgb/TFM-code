#!/bin/bash

echo "Training of GPT2 on Medical Data."

echo "Unzipping data..."
# unzip data
unzip data.zip

echo "Preparing folder structure"
# generate folder structure
mkdir trained_models
mkdir results

echo "Installing dependencies..."
# install dependencies
pip3 install -r requirements.txt --quiet

echo "Ready. Let's train."
# run model on data
python3 src/gptorch.py

echo "Finished."
echo "Zipping back the results..."
# zip all back
zip results.zip trained_models results 

echo "Program finished successfully."