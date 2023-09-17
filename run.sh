#!/bin/bash

# Usage: bash run.sh <gpu_id>

GPU=$1

# Benchmark on tree sizes in range(NMIN, NMAX, STEP), and for each size average values over RUNS runs
NMIN=5
NMAX=75
STEP=5
RUNS=100

export CUDA_VISIBLE_DEVICES=$GPU
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Running Sinkhorn:"
python3 main2.py sinkhorn $NMIN $NMAX $STEP $RUNS

# echo "Running CTreeOT:"
# python3 main.py ctreeot $NMIN $NMAX $STEP $RUNS
