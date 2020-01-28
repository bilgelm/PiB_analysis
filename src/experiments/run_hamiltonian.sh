#!/bin/bash

# main args
CURRENT_PATH=$(PWD)
PYTHON_SCRIPT='main_hamiltonian.py'
OUTPUTDIR='/Users/paul.vernhet/PhD_Aramis/experiments/results/hamiltonian'
DATADIR='/Users/paul.vernhet/PhD_Aramis/experiments/data/ode_data/wrap_summer2019'

# run FIRST ORDER script (no GPU) | no standardization | with filter | Linear Kernel | pre-tuning of kernel scale
python ${CURRENT_PATH}/${PYTHON_SCRIPT} --output_dir ${OUTPUTDIR} --data_dir ${DATADIR} \
--min_visits 3 --batch_size 16 --filter_quantiles --eps_quantiles .15 \
--model_type 'LINEAR' --tuning 5 \
--method 'midpoint' --epochs 100 \
--use_vars

# --preprocessing