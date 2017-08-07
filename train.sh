#!/bin/bash

unset OMP_NUM_THREADS
export OMP_NUM_THREADS=68
export MKL_NUM_THREADS=68
export KMP_AFFINITY=compact,1,0,granularity=fine

python ./train/train.py

