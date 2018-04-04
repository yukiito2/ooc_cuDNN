
#include "ooc_cudnn.h"
#include "ooc_cublas.h"

#include <sys/time.h>


cublasStatus_t CUBLASWINAPI ooc_cublasSgemm_optimize(
                                const size_t                        m,
                                const size_t                        n,
                                const size_t                        k,
                                const bool                          A_on_device,
                                const bool                          B_on_device,
                                const bool                          C_on_device,
                                const size_t                        data_size,
                                int                                *m_step,
                                int                                *n_step,
                                int                                *k_step );



cublasStatus_t CUBLASWINAPI ooc_cublasSgemm(
                                cublasHandle_t                      handle, 
                                cublasOperation_t                   transa, 
                                cublasOperation_t                   transb, 
                                int                                 m, 
                                int                                 n, 
                                int                                 k, 
                                const float                        *alpha, 
                                const float                        *A, 
                                int                                 lda, 
                                const float                        *B, 
                                int                                 ldb, 
                                const float                        *beta, 
                                float                              *C, 
                                int                                 ldc ){


        cublasStatus_t status;

        cudaPointerAttributes A_attr;
        cudaPointerAttributes B_attr;
        cudaPointerAttributes C_attr;

        cudaPointerGetAttributes(&A_attr, A);
        cudaPointerGetAttributes(&B_attr, B);
        cudaPointerGetAttributes(&C_attr, C);

        bool A_on_device = (A_attr.memoryType == cudaMemoryTypeDevice);
        bool B_on_device = (B_attr.memoryType == cudaMemoryTypeDevice);
        bool C_on_device = (C_attr.memoryType == cudaMemoryTypeDevice);

        int i, j;

        // initialize stream 
        cudaStream_t stream_handle;
        cublasGetStream(handle, &stream_handle);

        int current = 0;

        int low_priority, high_priority;
        cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority);

        cudaStream_t streams[3];
        streams[0] = stream_handle;
        for(i = 1; i < 3; i++) cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, low_priority);

        cudaStream_t streams_D2H[3];
        for(i = 0; i < 3; i++) cudaStreamCreateWithPriority(&streams_D2H[i], cudaStreamNonBlocking, high_priority);

        cudaEvent_t flags[3];
        for(i = 0; i < 3; i++){
                cudaEventCreate(&flags[i]);
                cudaEventRecord(flags[i], streams[i]);
        }
        
        

        // get parameter
        cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
        size_t data_size = cudnnSizeOf(dataType);

        
        // optimize
        int m_step, k_step, n_step;
        int num_pipeline = 1;

        ooc_cublasSgemm_optimize(m, n, k, A_on_device, B_on_device, C_on_device, data_size, &m_step, &n_step, &k_step);

        if(m_step * n_step * k_step > 1){
            if(m_step * n_step * k_step == 2) num_pipeline = 2;
            else num_pipeline = 3;
        }

        int d_m = m / m_step; if(0 < m % m_step) d_m++;
        int d_k = k / k_step; if(0 < k % k_step) d_k++;
        int d_n = n / n_step; if(0 < n % n_step) d_n++;


        // initialize d_A
        void *d_A[3];
        size_t d_A_size = d_m * d_k * data_size;

        if(!A_on_device){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_A[i], d_A_size);
        }


        // initialize d_B
        void *d_B[3];
        size_t d_B_size = d_k * d_n * data_size;

        if(!B_on_device){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_B[i], d_B_size);
        }


        // initialize d_C
        void *d_C[3];
        size_t d_C_size = d_m * d_n * data_size;

        if(!C_on_device){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_C[i], d_C_size);
        }


        // sgemm loop
        int i_m = 0, i_k = 0, i_n = 0;
        int d_lda = lda, d_ldb = ldb, d_ldc = ldc;

        int m_count, k_count, n_count;

        float one = 1.0f;
        float zero = 0.0f;
        float *beta_ptr;

        // for get (or set) sub Tensor
        cudnnHandle_t cudnn_handle;
        cudnnCreate(&cudnn_handle);
        cudnnSetStream(cudnn_handle, streams[current]);

        cudnnTensorDescriptor_t srcDesc, dstDesc;
        cudnnCreateTensorDescriptor(&srcDesc);
        cudnnCreateTensorDescriptor(&dstDesc);


        // initialize m_count
        m_count = 0;

        for(i_m = 0; i_m < m_step; i_m++){

                d_m = m / m_step;
                if(i_m < m % m_step) d_m++;

                // initialize n_count
                n_count = 0;

                for(i_n = 0; i_n < n_step; i_n++){

                        d_n = n / n_step;
                        if(i_n < n % n_step) d_n++;

                        // update d_C
                        if(C_on_device) d_C[current] = C + (m_count * n + n_count) * data_size;
                        else d_ldc = d_m;


                        k_count = 0;

                        for(i_k = 0; i_k < k_step; i_k++){

                                d_k = k / k_step;
                                if(i_k < k % k_step) d_k++;

                                // update d_A
                                if(A_on_device){
                                        if(transb == CUBLAS_OP_N) d_A[current] = (void *)A + (m_count * k + k_count) * data_size;
                                        else d_A[current] = (void *)A + (k_count * m + m_count) * data_size;
                                }
                                else{
                                        if(transa == CUBLAS_OP_N){
                                                cudnnSetTensor4dDescriptor(srcDesc, CUDNN_TENSOR_NCHW, dataType, 1, 1, m, k);
                                                cudnnSetTensor4dDescriptor(dstDesc, CUDNN_TENSOR_NCHW, dataType, 1, 1, d_m, d_k);
                                                ooc_cudnnGetSubTensor4D(cudnn_handle, srcDesc, A, dstDesc, d_A[current], 0, 0, m_count, k_count);
                                                d_lda = d_m;
                                        }
                                        else{
                                                cudnnSetTensor4dDescriptor(srcDesc, CUDNN_TENSOR_NCHW, dataType, 1, 1, k, m);
                                                cudnnSetTensor4dDescriptor(dstDesc, CUDNN_TENSOR_NCHW, dataType, 1, 1, d_k, d_m);
                                                ooc_cudnnGetSubTensor4D(cudnn_handle, srcDesc, A, dstDesc, d_A[current], 0, 0, k_count, m_count);
                                                d_lda = d_k;
                                        }
                                }


                                // update d_B
                                if(B_on_device){
                                        if(transb == CUBLAS_OP_N) d_B[current] = (void *)B + (k_count * n + n_count) * data_size;
                                        else d_B[current] = (void *)B + (n_count * k + k_count) * data_size;
                                }
                                else{
                                        if(transb == CUBLAS_OP_N){
                                                cudnnSetTensor4dDescriptor(srcDesc, CUDNN_TENSOR_NCHW, dataType, 1, 1, k, n);
                                                cudnnSetTensor4dDescriptor(dstDesc, CUDNN_TENSOR_NCHW, dataType, 1, 1, d_k, d_n);
                                                ooc_cudnnGetSubTensor4D(cudnn_handle, srcDesc, B, dstDesc, d_B[current], 0, 0, k_count, n_count);
                                                d_ldb = d_k;
                                        }
                                        else{
                                                cudnnSetTensor4dDescriptor(srcDesc, CUDNN_TENSOR_NCHW, dataType, 1, 1, n, k);
                                                cudnnSetTensor4dDescriptor(dstDesc, CUDNN_TENSOR_NCHW, dataType, 1, 1, d_n, d_k);
                                                ooc_cudnnGetSubTensor4D(cudnn_handle, srcDesc, B, dstDesc, d_B[current], 0, 0, n_count, k_count);
                                                d_ldb = d_n;
                                        }
                                }


                                cudaStreamWaitEvent(streams[current], flags[current], 0);


                                // sgemm main
                                if(i_k == 0) beta_ptr = &zero;
                                else  beta_ptr = &one;
                                cublasSgemm(handle, transa, transb, d_m, d_n, d_k, alpha, 
                                            (float *)d_A[current], d_lda, (float *)d_B[current], d_ldb, beta_ptr, (float *)d_C[current], d_ldc);


                                k_count += d_k;
                        }

                        cudaEventRecord(flags[current], streams[current]);
                        cublasSetStream(handle, streams_D2H[current]);
                        cudnnSetStream(cudnn_handle, streams_D2H[current]);
                        cudaStreamWaitEvent(streams_D2H[current], flags[current], 0);
                        cudaStreamWaitEvent(streams_D2H[current], flags[(current + num_pipeline - 1) % num_pipeline], 0);


                        // update h_C
                        if(!C_on_device){
                                cudnnSetTensor4dDescriptor(srcDesc, CUDNN_TENSOR_NCHW, dataType, 1, 1, d_m, d_n);
                                cudnnSetTensor4dDescriptor(dstDesc, CUDNN_TENSOR_NCHW, dataType, 1, 1, m, n);
                                ooc_cudnnSetSubTensor4D(cudnn_handle, dstDesc, C, srcDesc, d_C[current], 0, 0, m_count, n_count);
                        }

                        cudaEventRecord(flags[current], streams_D2H[current]);


                        current = (current + 1) % num_pipeline;
                        cublasSetStream(handle, streams[current]);
                        cudnnSetStream(cudnn_handle, streams[current]);


                        n_count += d_n;
                }


                m_count += d_m;
        }
        

        // finalize stream 
        cublasSetStream(handle, stream_handle);
        for(i = 0; i < 3; i++){
                cudaStreamSynchronize(streams[i]);
                cudaStreamSynchronize(streams_D2H[i]);
        }

        cudnnDestroy(cudnn_handle);
        cudnnDestroyTensorDescriptor(srcDesc);
        cudnnDestroyTensorDescriptor(dstDesc);

        if(!A_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_A[i]);
        if(!B_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_B[i]);
        if(!C_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_C[i]);

        
        return status;
}
