
#include "ooc_cudnn.h"


cudnnStatus_t CUDNNWINAPI ooc_cudnnActivationForward_optimize(
                                const size_t                        n,
                                const size_t                        c,
                                const size_t                        h,
                                const size_t                        w,
                                const bool                          x_on_device,
                                const bool                          y_on_device, 
                                const size_t                        data_size,
                                int                                *n_step,
                                int                                *c_step,
                                int                                *h_step );



cudnnStatus_t CUDNNWINAPI ooc_cudnnActivationForward(
                                cudnnHandle_t                       handle,
                                cudnnActivationDescriptor_t         activationDesc,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y ){

        cudnnStatus_t status;

        cudaPointerAttributes x_attr;
        cudaPointerAttributes y_attr;

        cudaPointerGetAttributes(&x_attr, x);
        cudaPointerGetAttributes(&y_attr, y);

        bool x_on_device = (x_attr.memoryType == cudaMemoryTypeDevice);
        bool y_on_device = (y_attr.memoryType == cudaMemoryTypeDevice);


        int i, j;

        // initialize stream 
        cudaStream_t stream_handle;
        cudnnGetStream(handle, &stream_handle);

        int current = 0;

        int low_priority, high_priority;
        cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority);

        cudaStream_t streams[3];
        streams[0] = stream_handle;
        for(i = 1; i < 3; i++) cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, low_priority);

        cudaEvent_t flags_H2D[3];
        for(i = 0; i < 3; i++){
                cudaEventCreate(&flags_H2D[i]);
                cudaEventRecord(flags_H2D[i], streams[i]);
        }

        cudaEvent_t flags_D2H[3];
        for(i = 0; i < 3; i++){
                cudaEventCreate(&flags_D2H[i]);
                cudaEventRecord(flags_D2H[i], streams[i]);
        }
        
        

        // get parameter
        cudnnDataType_t x_dataType;
        cudnnTensorFormat_t x_format = CUDNN_TENSOR_NCHW;
        int n, c, h, w, x_nStride_tmp, x_cStride_tmp, x_hStride_tmp, x_wStride_tmp;
        cudnnGetTensor4dDescriptor(xDesc, &x_dataType, &n, &c, &h, &w, &x_nStride_tmp, &x_cStride_tmp, &x_hStride_tmp, &x_wStride_tmp);
        size_t x_nStride = (size_t)c * (size_t)h * (size_t)w, x_cStride = (size_t)h * (size_t)w, x_hStride = (size_t)w, x_wStride = 1;

        cudnnDataType_t y_dataType;
        cudnnTensorFormat_t y_format = CUDNN_TENSOR_NCHW;
        int y_n, y_c, y_h, y_w, y_nStride_tmp, y_cStride_tmp, y_hStride_tmp, y_wStride_tmp;
        cudnnGetTensor4dDescriptor(yDesc, &y_dataType, &y_n, &y_c, &y_h, &y_w, &y_nStride_tmp, &y_cStride_tmp, &y_hStride_tmp, &y_wStride_tmp);
        size_t y_nStride = (size_t)y_c * (size_t)y_h * (size_t)y_w, y_cStride = (size_t)y_h * (size_t)y_w, y_hStride = (size_t)y_w, y_wStride = 1;
    
        size_t y_data_size = cudnnSizeOf(y_dataType);
        size_t x_data_size = cudnnSizeOf(x_dataType);


        // check parameter 
        if((n != y_n) || (c != y_c) || (h != y_h)){
                printf("ooc_cudnnActivationForward : CUDNN_STATUS_BAD_PARAM\n");
                printf("%d, %d\n", n, y_n);
                printf("%d, %d\n", c, y_c);
                printf("%d, %d\n", h, y_h);
                printf("%d, %d\n", w, y_w);
                return CUDNN_STATUS_BAD_PARAM;
        }
        if((y_format != CUDNN_TENSOR_NCHW) || (x_format != CUDNN_TENSOR_NCHW)){
                printf("ooc_cudnnActivationForward : CUDNN_STATUS_NOT_SUPPORTED\n");
                return CUDNN_STATUS_NOT_SUPPORTED;
        }

        
        // optimize
        int n_step, c_step, h_step;
        int num_pipeline = 1;

        ooc_cudnnActivationForward_optimize(n, c, h, w, x_on_device, y_on_device, y_data_size, &n_step, &c_step, &h_step);

        if(n_step * c_step * h_step > 1){
            if(n_step * c_step * h_step == 2) num_pipeline = 2;
            else num_pipeline = 3;
        }

        int d_n = y_n / n_step; if(0 < y_n % n_step) d_n++;
        int d_c = y_c / c_step; if(0 < y_c % c_step) d_c++;
        int d_h = h / h_step; if(0 < h % h_step) d_h++;


        // initialize d_x
        void *d_x[3];
        size_t d_x_size = d_n * d_c * d_h * w * x_data_size;

        if(!x_on_device){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_x[i], d_x_size);
        }

        cudnnTensorDescriptor_t d_xDesc;
        cudnnCreateTensorDescriptor(&d_xDesc);


        // initialize d_y
        void *d_y[3];
        size_t d_y_size = d_n * d_c * d_h * w * y_data_size;

        if(!y_on_device){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_y[i], d_y_size);
        }

        cudnnTensorDescriptor_t d_yDesc;
        cudnnCreateTensorDescriptor(&d_yDesc);


        // activation loop
        int i_n, i_c, i_h;

        int n_count, c_count, h_count;


        // initialize n_count
        n_count = 0;

        for(i_n = 0; i_n < n_step; i_n++){

                d_n = n / n_step;
                if(i_n < n % n_step) d_n++;

                // initialize c_count
                c_count = 0;

                for(i_c = 0; i_c < c_step; i_c++){

                        d_c = c / c_step;
                        if(i_c < c % c_step) d_c++;

                        // initialize h_count
                        h_count = 0;

                        for(i_h = 0; i_h < h_step; i_h++){

                                d_h = h / h_step;
                                if(i_h < h % h_step) d_h++;

                                cudaStreamWaitEvent(streams[current], flags_H2D[(current + num_pipeline - 1) % num_pipeline], 0);


                                // update d_x
                                if(x_on_device){
                                        cudnnSetTensor4dDescriptorEx(d_xDesc, x_dataType, d_n, d_c, d_h, w, x_nStride, x_cStride, x_hStride, x_wStride);
                                        d_x[current] = (void *)x + (n_count * x_nStride + c_count * x_cStride + h_count * x_hStride) * x_data_size;
                                }
                                else{
                                        cudnnSetTensor4dDescriptor(d_xDesc, x_format, x_dataType, d_n, d_c, d_h, w);
                                        ooc_cudnnGetSubTensor4D(handle, xDesc, x, d_xDesc, d_x[current], n_count, c_count, h_count, 0);
                                }


                                // update d_y
                                if(y_on_device){
                                        cudnnSetTensor4dDescriptorEx(d_yDesc, y_dataType, d_n, d_c, d_h, w, y_nStride, y_cStride, y_hStride, y_wStride);
                                        d_y[current] = y + (n_count * y_nStride + c_count * y_cStride + h_count * y_hStride) * y_data_size;
                                }
                                else{
                                        cudnnSetTensor4dDescriptor(d_yDesc, y_format, y_dataType, d_n, d_c, d_h, w);
                                }


                                cudaEventRecord(flags_H2D[current], streams[current]);


                                // activation main
                                status = cudnnActivationForward(handle, activationDesc, alpha, d_xDesc, d_x[current], beta, d_yDesc, d_y[current]);


                                cudaStreamWaitEvent(streams[current], flags_D2H[(current + num_pipeline - 1) % num_pipeline], 0);

                                // update h_y
                                if(!y_on_device){
                                        ooc_cudnnSetSubTensor4D(handle, yDesc, y, d_yDesc, d_y[current], n_count, c_count, h_count, 0);
                                }

                                cudaEventRecord(flags_D2H[current], streams[current]);


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

        cudnnDestroyTensorDescriptor(d_xDesc);
        cudnnDestroyTensorDescriptor(d_yDesc);


        if(!x_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_x[i]);
        if(!y_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_y[i]);

        return status;
}

