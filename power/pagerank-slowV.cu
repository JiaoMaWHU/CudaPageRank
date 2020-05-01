#include "pagerank.h"
#include "matrix_operations.h"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <climits>
using namespace std;

int read_file(unordered_map<int, vector<int> > &m, string filename){
    ifstream infile("../data/"+filename);
    string line;
    int from, to;
    int n = INT_MIN, edge = 0;
    while(getline(infile, line)) {
        stringstream ss(line);
        ss >> from >> to;
        n = max(n, max(from, to));
        n = min(999, n);
        if(from<1000 && to<1000){
            m[from].push_back(to);
            edge++;
        }
    }
    printf("Matrix stat: num of node = %d, num of edge = %d.\n", n+1, edge);
    return n+1;
}

int run_pagerank(string filename, float epsilon){
    // read file
    unordered_map<int, vector<int> > mat;
    size_t node_size = read_file(mat, filename);

    // build m and M matrix
    float m[node_size*node_size]{};
    int size = 0; 
    float value = 0;
    for(auto& pair: mat){
        size = pair.second.size();
        value = 1.0/size;
        for(int& id: pair.second){
            m[id*node_size+pair.first] = value;
        }
    }

    // init cuda matrixes
    float *M, *A;
    float *y, *X1, *X2;
    cudaMalloc((void **)&M, sizeof(float)*node_size*node_size);
    cudaMalloc((void **)&A, sizeof(float)*node_size*node_size);
    cudaMalloc((void **)&y, sizeof(float)*node_size);
    cudaMalloc((void **)&X1, sizeof(float)*node_size);
    cudaMalloc((void **)&X2, sizeof(float)*node_size);
    
    // build start matrix A, M->tmp use
    cudaMemcpy(M, m, sizeof(float)*node_size*node_size, cudaMemcpyHostToDevice);
    float d = 0.85;
    value_mul_matrix<<<(node_size*node_size+1023)/1024,1024>>>(M, A, node_size, node_size, d);
    value_add_matrix<<<(node_size*node_size+1023)/1024,1024>>>(A, A, node_size, node_size, (1-d)/node_size);
    
    // compute, do it in reachiability way
    cudaMemset(X1, 0, sizeof(float)*node_size); // init to zero
    value_add_matrix<<<(node_size*node_size+1023)/1024,1024>>>(X1, X1, node_size, 1, 1.0);
    unsigned int iter = 0;
    int* max_norm_int = 0;
    float* max_norm_float = 0;
    float iter_eplson = 0;
    cudaMalloc((void **)&max_norm_int, sizeof(int));
    cudaMalloc((void **)&max_norm_float, sizeof(float));
    float* A1 = A, *A2 = M;
    float* xt = X1, *xt_p_1 = X2; 
    dim3 dim_block(32, 32, 1);

    unsigned long long start = 
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    do{
        iter++;

        // A*A
        // dim3 dim_grid((node_size-1)/32+1, (node_size-1)/32+1, 1);
        // matrix_mul_matrix<<<dim_grid, dim_block>>>(A1, A1, A2, node_size, node_size, node_size, node_size);
        
        // A*x
        dim3 dim_grid1((1-1)/32+1, (node_size-1)/32+1, 1);
        matrix_mul_matrix<<<dim_grid1, dim_block>>>(A1, xt, y, node_size, 1, node_size, 1);

        // x = y/y
        max_norm_matrix<<<(node_size*1+1023)/1024, 1024>>>(y, node_size, 1, max_norm_int, max_norm_float);
        // seq_max_norm<<<1,1>>>(y, node_size, 1, max_norm);
        cudaMemcpy(&iter_eplson, max_norm_float, sizeof(float), cudaMemcpyDeviceToHost);
        value_mul_matrix<<<(node_size*1+1023)/1024,1024>>>(y, xt_p_1, node_size, 1, 1.0/iter_eplson);
        
        // x-x
        matrix_add_matrix<<<(node_size*node_size+1023)/1024,1024>>>(xt_p_1, xt, xt, node_size, 1, -1);
        max_norm_matrix<<<(node_size*1+1023)/1024, 1024>>>(xt, node_size, 1, max_norm_int, max_norm_float);
        // seq_max_norm<<<1,1>>>(xt, node_size, 1, max_norm);
        cudaMemcpy(&iter_eplson, max_norm_float, sizeof(float), cudaMemcpyDeviceToHost);

        // printf("Inter no.%d, difference: %0.9f\n", iter, iter_eplson);
        // swap(A1, A2);
        swap(xt_p_1, xt);

    }while(iter_eplson>=epsilon && iter<100);
    unsigned long long after = 
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    after = after - start;
    printf("Difference: %0.9f, total iter: %d, used time in ms: %d\n", iter_eplson, iter, after);
    
    // print normalized pagerank value for each node
    float* res = new float[node_size];
    cudaMemcpy(res, xt, node_size*sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0;
    for (int i = 0; i < node_size; i++) sum += res[i];
    for (int i = 0; i < node_size; i++) res[i] /= sum;
    for (int i = 0; i < node_size; i++) printf("PageRank for Node %d: %0.9f\n", i, res[i]);
    delete(res);

    // free
    cudaFree(M);
    cudaFree(A);
    cudaFree(y);
    cudaFree(X1);
    cudaFree(X2);
    cudaFree(max_norm_int);
    cudaFree(max_norm_float);
    
    return iter_eplson;
}