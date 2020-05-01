#pragma once

__global__ void print_mat(float* mat, int row, int col);

__global__ void value_mul_matrix(float* mat1, float* mat2, int row, int col, float v);

__global__ void value_add_matrix(float* mat1, float* mat2, int row, int col, float v);

__global__ void matrix_mul_matrix(float *A, float *B, float *C, int col_A, int col_B, int row_C, int col_C);

__global__ void matrix_add_matrix(float* mat1, float* mat2, float* mat3, int row, int col, int sign);

__global__ void max_norm_matrix(float* mat1, int row, int col, int* norm, float* final_norm);

__global__ void seq_max_norm(float* mat1, int row, int col, float* norm);

