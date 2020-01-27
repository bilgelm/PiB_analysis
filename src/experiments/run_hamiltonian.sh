#!/bin/bash

# main args
CURRENT_PATH=$(pwd)
PYTHON_SCRIPT='main_hamiltonian.py'
OUTPUTDIR='/cog/murat/for_paulvernhet/results/hamiltonian'
DATADIR='/cog/murat/for_bruno/monotone_reg/data'

# run FIRST ORDER script (no GPU) | no standardization | with filter | Linear Kernel | pre-tuning of kernel scale
python ${CURRENT_PATH}/${PYTHON_SCRIPT} --output_dir ${OUTPUTDIR} --data_dir ${DATADIR} \
--min_visits 3 --batch_size 16 --filter_quantiles --eps_quantiles .15 --preprocessing \
--model_type 'LINEAR' --tuning 5 \
--method 'midpoint' --epochs 100 \
--pib_threshold 1.0571
