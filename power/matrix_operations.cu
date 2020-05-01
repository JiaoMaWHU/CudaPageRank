#include "matrix_operations.h"
#include <stdio.h>

__global__ void print_mat(float* mat, int row, int col){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id==0){
        for(int i=0; i<row; i++){
            for(int j =0; j<col; j++)
                printf("%0.3f\t", mat[i*col+j]);
            printf("\n");
        }  
        printf("\n");
    }
}

__global__ void value_mul_matrix(float* mat1, float* mat2, int row, int col, float v){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = row*col;
    if(id<size){
        mat2[id] = mat1[id] * v;
    }
}

__global__ void value_add_matrix(float* mat1, float* mat2, int row, int col, float v){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = row*col;
    if(id<size){
        mat2[id] = mat1[id] + v;
    }
}

__global__ void matrix_mul_matrix(float *A, float *B, float *C, int col_A, int col_B, int row_C, int col_C){
    float sum = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < row_C && col < col_C) {
        for (int i = 0; i < col_A; ++i) {
            sum += A[row * col_A + i] * B[i * col_B + col];
        }
        C[row * col_B + col] = sum;
    }
}

__global__ void matrix_add_matrix(float* mat1, float* mat2, float* mat3, int row, int col, int sign){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = row*col;
    if(id<size){
        mat3[id] = mat1[id] + sign*mat2[id];
    }
}


__global__ void max_norm_matrix(float* mat1, int row, int col, int* norm, float* final_norm){
    *norm = 0;
    __syncthreads();
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = row*col;
    if(id<size){
        atomicMax(norm, __float_as_int(abs(mat1[id])));
    }
    __syncthreads();
    if(id==0){
        *final_norm = __int_as_float(*norm);
    }
}

__global__ void seq_max_norm(float* mat1, int row, int col, float* norm){
    *norm = 0;
    for(int i=0; i<row; i++){
        for(int j =0; j<col; j++)
            *norm = max(abs(mat1[i*col+j]), *norm);
    }  
}