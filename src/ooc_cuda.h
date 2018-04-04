
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <string.h>

cudaError_t ooc_cudaMemcpyProfile();

cudaError_t ooc_cudaMalloc(void **devPtr, size_t size);

cudaError_t ooc_cudaFree(void *devPtr);

cudaError_t ooc_cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);

cudaError_t ooc_cudaMemset(void *devPtr, int value, size_t count);