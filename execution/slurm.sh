#!/bin/bash

#SBATCH --job-name GPT2-TFM # Nombre del proceso
#SBATCH --partition dios # Cola para ejecutar
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"

conda create --prefix pt160py37 python=3.7
conda activate pt160py37
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch

bash executor.sh



mail -s "Proceso finalizado" blograso@gmail.com <<< "El proceso ha finalizado"