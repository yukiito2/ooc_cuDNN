
#include "ooc_cudnn.h"


cudnnStatus_t CUDNNWINAPI ooc_cudnnAddTensor_optimize(
                                const size_t                        n,
                                const size_t                        c,
                                const size_t                        h,
                                const size_t                        w,
                                const bool                          A_on_device,
                                const bool                          C_on_device, 
                                const size_t                        data_size,
                                int                                *n_step,
                                int                                *c_step,
                                int                                *h_step );



// 現在bias用
cudnnStatus_t CUDNNWINAPI ooc_cudnnAddTensor(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       ADesc,
                                const void                         *A,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       CDesc,
                                void                               *C ){

        int i, j;
        cudnnStatus_t status;

    	cudaPointerAttributes A_attr;
    	cudaPointerAttributes C_attr;

    	cudaPointerGetAttributes(&A_attr, A);
    	cudaPointerGetAttributes(&C_attr, C);

    	bool A_on_device = (A_attr.memoryType == cudaMemoryTypeDevice);
        bool C_on_device = (C_attr.memoryType == cudaMemoryTypeDevice);

    
        // initialize stream 
        cudaStream_t stream_handle;
        cudnnGetStream(handle, &stream_handle);

        int current = 0;

        cudaStream_t streams[3];
        streams[0] = stream_handle;
        for(i = 1; i < 3; i++) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        
        

        // get parameter
        cudnnDataType_t A_dataType;
        cudnnTensorFormat_t A_format = CUDNN_TENSOR_NCHW;
        int A_n, A_c, A_h, A_w, A_nStride_tmp, A_cStride_tmp, A_hStride_tmp, A_wStride_tmp;
        cudnnGetTensor4dDescriptor(ADesc, &A_dataType, &A_n, &A_c, &A_h, &A_w, &A_nStride_tmp, &A_cStride_tmp, &A_hStride_tmp, &A_wStride_tmp);
        size_t A_nStride = (size_t)A_c * (size_t)A_h * (size_t)A_w, A_cStride = (size_t)A_h * (size_t)A_w, A_hStride = (size_t)A_w, A_wStride = 1;

        cudnnDataType_t C_dataType;
        cudnnTensorFormat_t C_format = CUDNN_TENSOR_NCHW;
        int C_n, C_c, C_h, C_w, C_nStride_tmp, C_cStride_tmp, C_hStride_tmp, C_wStride_tmp;
        cudnnGetTensor4dDescriptor(CDesc, &C_dataType, &C_n, &C_c, &C_h, &C_w, &C_nStride_tmp, &C_cStride_tmp, &C_hStride_tmp, &C_wStride_tmp);
        size_t C_nStride = (size_t)C_c * (size_t)C_h * (size_t)C_w, C_cStride = (size_t)C_h * (size_t)C_w, C_hStride = (size_t)C_w, C_wStride = 1;
    
        size_t C_data_size = cudnnSizeOf(C_dataType);
        size_t A_data_size = cudnnSizeOf(A_dataType);


        // check parameter 
        if((A_n != 1) || (A_c != C_c) || (A_h != 1) || (A_w != 1)){
                printf("ooc_cudnnAddTensor : CUDNN_STATUS_BAD_PARAM\n");
                printf("%d, %d\n", A_n, C_n);
                printf("%d, %d\n", A_c, C_c);
                printf("%d, %d\n", A_h, C_h);
                printf("%d, %d\n", A_w, C_w);
                return CUDNN_STATUS_BAD_PARAM;
        }
        if((C_format != CUDNN_TENSOR_NCHW) || (A_format != CUDNN_TENSOR_NCHW)){
                printf("ooc_cudnnAddTensor : CUDNN_STATUS_NOT_SUPPORTED\n");
                return CUDNN_STATUS_NOT_SUPPORTED;
        }

        
        // optimize
        int n_step, c_step, h_step;
        int num_pipeline = 1;

        ooc_cudnnAddTensor_optimize(C_n, C_c, C_h, C_w,  A_on_device, C_on_device, C_data_size, &n_step, &c_step, &h_step);

        if(n_step * c_step * h_step > 1){
            if(n_step * c_step * h_step == 2) num_pipeline = 2;
            else num_pipeline = 3;
        }

        int d_n = C_n / n_step; if(0 < C_n % n_step) d_n++;
        int d_c = C_c / c_step; if(0 < C_c % c_step) d_c++;
        int d_h = C_h / h_step; if(0 < C_h % h_step) d_h++;

        // initialize d_A
        void *d_A[3];
        size_t d_A_size = 1 * d_c * 1 * 1 * A_data_size;

        if(!A_on_device){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_A[i], d_A_size);
        }

        cudnnTensorDescriptor_t d_ADesc;
        cudnnCreateTensorDescriptor(&d_ADesc);


        // initialize d_C
        void *d_C[3];
        size_t d_C_size = d_n * d_c * d_h * C_w * C_data_size;

        if(!C_on_device){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_C[i], d_C_size);
        }

        cudnnTensorDescriptor_t d_CDesc;
        cudnnCreateTensorDescriptor(&d_CDesc);


        // addTensor loop
        int i_n, i_c, i_h;

        int n_count, c_count, h_count;


        // initialize n_count
        n_count = 0;

        for(i_n = 0; i_n < n_step; i_n++){

                d_n = C_n / n_step;
                if(i_n < C_n % n_step) d_n++;

                // initialize c_count
                c_count = 0;

                for(i_c = 0; i_c < c_step; i_c++){

                        d_c = C_c / c_step;
                        if(i_c < C_c % c_step) d_c++;

                        // initialize h_count
                        h_count = 0;

                        for(i_h = 0; i_h < h_step; i_h++){

                                d_h = C_h / h_step;
                                if(i_h < C_h % h_step) d_h++;

                                // update d_A
                                if(A_on_device){
                                        cudnnSetTensor4dDescriptor(d_ADesc, CUDNN_TENSOR_NCHW, A_dataType, 1, d_c, 1, 1);
                                        d_A[current] = (void *)A + (c_count * A_cStride) * A_data_size;
                                }
                                else{
                                        cudnnSetTensor4dDescriptor(d_ADesc, A_format, A_dataType, 1, d_c, 1, 1);
                                        ooc_cudnnGetSubTensor4D(handle, ADesc, A, d_ADesc, d_A[current], 0, c_count, 0, 0);
                                }


                                // update d_C
                                if(C_on_device){
                                        cudnnSetTensor4dDescriptorEx(d_CDesc, C_dataType, d_n, d_c, d_h, C_w, C_nStride, C_cStride, C_hStride, C_wStride);
                                        d_C[current] = C + (n_count * C_nStride + c_count * C_cStride + h_count * C_hStride) * C_data_size;
                                }
                                else{
                                        cudnnSetTensor4dDescriptor(d_CDesc, C_format, C_dataType, d_n, d_c, d_h, C_w);
                                        ooc_cudnnGetSubTensor4D(handle, CDesc, C, d_CDesc, d_C[current], n_count, c_count, h_count, 0);
                                }


                                // addTensor main
                                status = cudnnAddTensor(handle, alpha, d_ADesc, d_A[current], beta, d_CDesc, d_C[current]);


                                // update h_C
                                if(!C_on_device){
                                        ooc_cudnnSetSubTensor4D(handle, CDesc, C, d_CDesc, d_C[current], n_count, c_count, h_count, 0);
                                }

                                h_count += d_h;


                                current = (current + 1) % num_pipeline;
                                cudnnSetStream(handle, streams[current]);
                        }


                        c_count += d_c;
                }


                n_count += d_n;
        }
        

        // finalize stream 
        cudnnSetStream(handle, stream_handle);
        for(i = 0; i < 3; i++){
                cudaStreamSynchronize(streams[i]);
        }

        cudnnDestroyTensorDescriptor(d_ADesc);
        cudnnDestroyTensorDescriptor(d_CDesc);


        if(!C_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_C[i]);
        if(!A_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_A[i]);

        return status;
}