
#include "ooc_cuda.h"
#include <map>

double cudaMemcpy_weight;
double cudaMemcpy_bias;

std::map<void*, size_t> map_memptr_size;
size_t d_mem_free, d_mem_total;
bool ooc_cudaMalloc_flag = false;

cudaError_t ooc_cudaMemcpyProfile(){

        int i, loop = 1000;
        struct timeval t1, t2;
        double time1, time2;
        double size1, size2;

        unsigned long long int n = 1, c = 64, h = 256, w = 256;

        float *C;
        cudaMalloc((void**)&C, n*c*h*w*sizeof(float));


        // profile memcpy
        float *h_C;
        cudaMallocHost((void**)&h_C, n*c*h*w*sizeof(float));

        // time1, size1
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudaMemcpy(C, h_C, n*c*h*w*sizeof(float), cudaMemcpyDefault);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time1 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size1 = (double)(n * c * h * w * sizeof(float));


        // time2, size2
        h /= 2; w /= 2;

        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudaMemcpy(C, h_C, n*c*h*w*sizeof(float), cudaMemcpyDefault);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time2 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size2 = (double)(n * c * h * w * sizeof(float));

        // cudaMemcpy_weight, cudaMemcpy_bias
        cudaMemcpy_weight = (time1 - time2) / (size1 - size2);
        cudaMemcpy_bias = time1 - cudaMemcpy_weight * size1;
        cudaFree(C);
        cudaFreeHost(h_C);


        return cudaSuccess;
}



cudaError_t ooc_cudaMalloc (void **devPtr, size_t size){
        cudaError_t status;

        if(!ooc_cudaMalloc_flag){
                cudaMemGetInfo(&d_mem_free, &d_mem_total);
                d_mem_free = (size_t)((double)d_mem_free*0.9);
                if(d_mem_free < 0) d_mem_free = 0;
                ooc_cudaMalloc_flag = true;
        }

        if(size < d_mem_free){
                status = cudaMalloc(devPtr, size);
                map_memptr_size[*devPtr] = size;
                d_mem_free -= size;
        }
        else{
                status = cudaMallocHost(devPtr, size);
        }

        return status;
}


cudaError_t ooc_cudaFree(void *devPtr){
        cudaError_t status;

        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, devPtr);

        if(attr.memoryType == cudaMemoryTypeDevice){
                status = cudaFree(devPtr);
                size_t size = map_memptr_size[devPtr];
                map_memptr_size.erase(devPtr);
                d_mem_free += size;
        }
        else{
                status = cudaFreeHost(devPtr);
        }

        return status;

}


cudaError_t ooc_cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind){
        cudaError_t status;

        // debug
        cudaPointerAttributes src_attr;
        cudaPointerAttributes dst_attr;
        cudaPointerGetAttributes(&src_attr, src);
        cudaPointerGetAttributes(&dst_attr, dst);
        bool src_on_device = (src_attr.memoryType == cudaMemoryTypeDevice);
        bool dst_on_device = (dst_attr.memoryType == cudaMemoryTypeDevice);
        
        status = cudaMemcpy(dst, src, count, cudaMemcpyDefault);

        return status;
}



cudaError_t ooc_cudaMemset(void *devPtr, int value, size_t count){
        cudaError_t status;

        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, devPtr);

        if(attr.memoryType == cudaMemoryTypeDevice){
                status = cudaMemset(devPtr, value, count);
        }
        else{
                memset(devPtr, value, count);
                status = cudaSuccess;
        }

        return status;
}