#!/bin/bash

# main args
CURRENT_PATH=$(PWD)
PYTHON_SCRIPT='main_firstorder.py'
OUTPUTDIR='/Users/paul.vernhet/PhD_Aramis/experiments/results/firstorder'
DATADIR='/Users/paul.vernhet/PhD_Aramis/experiments/data/ode_data/wrap_summer2019'

python ${CURRENT_PATH}/${PYTHON_SCRIPT} --output_dir ${OUTPUTDIR} --data_dir ${DATADIR} \
--min_visits 3 --batch_size 16 --model_type 'POLY' --tuning 5 --preprocessing --use_vars