#pragma once

#include <Eigen/Dense>
#include "state.h"

using namespace Eigen;

template<typename T>
struct CudaVectorX {
    T* data;
};

template<typename T>
cudaError_t cudaMalloc(VectorX<T>* v, CudaVectorX<T>* cv) {
    return cudaMalloc((void**)&cv->data, v->size() * sizeof(T));
}

template<typename T>
cudaError_t memcpyHostToDevice(VectorX<T>* v, CudaVectorX<T>* cv) {
    return cudaMemcpy((void**)cv->data, v->data(), v->size() * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
cudaError_t memcpyDeviceToHost(VectorX<T>* v, CudaVectorX<T>* cv) {
    return cudaMemcpy((void**)v->data(), cv->data, v->size() * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
struct CudaMatrixX {
    T* data;
    size_t pitch;

    __device__ T* getRowPtr(int row) {
        return (T*)((char*)data + row*pitch);
    }    
};

template<typename T>
cudaError_t cudaMalloc(MatrixRX<T>* m, CudaMatrixX<T>* cm) {
   return cudaMallocPitch((void**)&cm->data, &cm->pitch, m->cols() * sizeof(T), m->rows());
}

template<typename T>
cudaError_t memcpyHostToDevice(MatrixRX<T>* m, CudaMatrixX<T>* cm) {
    return cudaMemcpy2D(cm->data, cm->pitch, m->data(), m->cols() * sizeof(T), m->cols() * sizeof(T), m->rows(), cudaMemcpyHostToDevice);
}

template<typename T>
cudaError_t memcpyDeviceToHost(MatrixRX<T>* m, CudaMatrixX<T>* cm) {
    return cudaMemcpy2D(m->data(), m->cols() * sizeof(T), cm->data, cm->pitch, m->cols() * sizeof(T), m->rows(), cudaMemcpyDeviceToHost);
}
