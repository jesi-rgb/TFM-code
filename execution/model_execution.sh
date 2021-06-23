#!/bin/bash

# unzip data
unzip data.zip

# generate folder structure
mkdir output
mkdir trained_models
mkdir results


# install dependencies
pip3 install -r requirements.txt

# run model on data

python3 gptorch.py


# zip all back
zip . results.zip