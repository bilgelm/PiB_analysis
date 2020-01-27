#!/bin/bash

# main args
CURRENT_PATH=$(pwd)
PYTHON_SCRIPT='main_firstorder.py'
OUTPUTDIR='/cog/murat/for_paulvernhet/results/firstorder'
DATADIR='/cog/murat/for_bruno/monotone_reg/data'

# run FIRST ORDER script (no GPU) | no standardization | Linear Kernel | pre-tuning of kernel scale
#python ${CURRENT_PATH}/${PYTHON_SCRIPT} --output_dir ${OUTPUTDIR} --data_dir ${DATADIR} \
#--min_visits 3 --batch_size 16 --model_type 'LINEAR' --tuning 25

# run FIRST ORDER script (no GPU) | with standardization | Polynomial Kernel | pre-tuning of kernel scale
python ${CURRENT_PATH}/${PYTHON_SCRIPT} --output_dir ${OUTPUTDIR} --data_dir ${DATADIR} \
--min_visits 3 --batch_size 16 --model_type 'POLY' --tuning 25 --preprocessing \
--pib_threshold 1.0571
