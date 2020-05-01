#!/bin/bash
nvcc main.cu pagerank.cu matrix_operations.cu -o main.o
sbatch job-batch