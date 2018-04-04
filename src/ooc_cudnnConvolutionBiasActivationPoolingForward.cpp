

#include "ooc_cudnn.h"

#include <sys/time.h>


cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBiasActivationPoolingForward_optimize(
                                const size_t                        n,
                                const size_t                        x_c,
                                const size_t                        x_h,
                                const size_t                        x_w,
                                const size_t                        w_h,
                                const size_t                        w_w,
                                const size_t                        y_c,
                                const size_t                        y_h,
                                const size_t                        y_w,
                                const size_t                        z_h,
                                const size_t                        z_w,
                                const bool                          x_on_device,
                                const bool                          w_on_device,
                                const bool                          b_on_device,
                                const bool                          y_on_device,
                                const bool                          z_on_device,
                                const size_t                        stride_h,
                                const size_t                        dilation_h,
                                const size_t                        pool_window_h,
                                const size_t                        pool_window_w,
                                const size_t                        pool_stride_h,
                                const size_t                        data_size,
                                int                                *n_step,
                                int                                *y_c_step,
                                int                                *x_c_step,
                                int                                *h_step );


// 現状の実装では、poolingのpad_h=0, window_h=stride_hでなければならない
cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBiasActivationPoolingForward(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionFwdAlgo_t           algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const cudnnTensorDescriptor_t       biasDesc,
                                const void                         *bias,
                                cudnnActivationDescriptor_t         activationDesc,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y,
                                cudnnPoolingDescriptor_t            poolingDesc,
                                const cudnnTensorDescriptor_t       zDesc,
                                void                               *z ){

        cudnnStatus_t status;

        cudaPointerAttributes x_attr;
        cudaPointerAttributes w_attr;
        cudaPointerAttributes y_attr;
        cudaPointerAttributes bias_attr;
        cudaPointerAttributes z_attr;

        cudaPointerGetAttributes(&x_attr, x);
        cudaPointerGetAttributes(&w_attr, w);
        cudaPointerGetAttributes(&y_attr, y);
        cudaPointerGetAttributes(&bias_attr, bias);
        cudaPointerGetAttributes(&z_attr, z);

        bool x_on_device = (x_attr.memoryType == cudaMemoryTypeDevice);
        bool w_on_device = (w_attr.memoryType == cudaMemoryTypeDevice);
        bool y_on_device = (y_attr.memoryType == cudaMemoryTypeDevice);
        bool bias_on_device = (bias_attr.memoryType == cudaMemoryTypeDevice);
        bool z_on_device = (z_attr.memoryType == cudaMemoryTypeDevice);

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

        cudaStream_t streams_D2H[3];
        for(i = 0; i < 3; i++) cudaStreamCreateWithPriority(&streams_D2H[i], cudaStreamNonBlocking, high_priority);

        cudaEvent_t flags[3];
        for(i = 0; i < 3; i++){
                cudaEventCreate(&flags[i]);
                cudaEventRecord(flags[i], streams[i]);
        }


        // get parameter
        cudnnDataType_t x_dataType;
        cudnnTensorFormat_t x_format = CUDNN_TENSOR_NCHW;
        int n, x_c, x_h, x_w, x_nStride_tmp, x_cStride_tmp, x_hStride_tmp, x_wStride_tmp;
        cudnnGetTensor4dDescriptor(xDesc, &x_dataType, &n, &x_c, &x_h, &x_w, &x_nStride_tmp, &x_cStride_tmp, &x_hStride_tmp, &x_wStride_tmp);
        size_t x_nStride = (size_t)x_c * (size_t)x_h * (size_t)x_w, x_cStride = (size_t)x_h * (size_t)x_w, x_hStride = (size_t)x_w, x_wStride = 1;

        cudnnDataType_t w_dataType;
        cudnnTensorFormat_t w_format;
        int w_k, w_c, w_h, w_w;
        cudnnGetFilter4dDescriptor(wDesc, &w_dataType, &w_format, &w_k, &w_c, &w_h, &w_w);

        cudnnDataType_t y_dataType;
        cudnnTensorFormat_t y_format = CUDNN_TENSOR_NCHW;
        int y_n, y_c, y_h, y_w, y_nStride_tmp, y_cStride_tmp, y_hStride_tmp, y_wStride_tmp;
        cudnnGetTensor4dDescriptor(yDesc, &y_dataType, &y_n, &y_c, &y_h, &y_w, &y_nStride_tmp, &y_cStride_tmp, &y_hStride_tmp, &y_wStride_tmp);
        size_t y_nStride = (size_t)y_c * (size_t)y_h * (size_t)y_w, y_cStride = (size_t)y_h * (size_t)y_w, y_hStride = (size_t)y_w, y_wStride = 1;

        cudnnDataType_t bias_dataType;
        cudnnTensorFormat_t bias_format = CUDNN_TENSOR_NCHW;
        int bias_n, bias_c, bias_h, bias_w, bias_nStride_tmp, bias_cStride_tmp, bias_hStride_tmp, bias_wStride_tmp;
        cudnnGetTensor4dDescriptor(biasDesc, &bias_dataType, &bias_n, &bias_c, &bias_h, &bias_w, &bias_nStride_tmp, &bias_cStride_tmp, &bias_hStride_tmp, &bias_wStride_tmp);
        size_t bias_nStride = (size_t)bias_c * (size_t)bias_h * (size_t)bias_w, bias_cStride = (size_t)bias_h * (size_t)bias_w, bias_hStride = (size_t)bias_w, bias_wStride = 1;

        cudnnDataType_t z_dataType;
        cudnnTensorFormat_t z_format = CUDNN_TENSOR_NCHW;
        int z_n, z_c, z_h, z_w, z_nStride_tmp, z_cStride_tmp, z_hStride_tmp, z_wStride_tmp;
        cudnnGetTensor4dDescriptor(zDesc, &z_dataType, &z_n, &z_c, &z_h, &z_w, &z_nStride_tmp, &z_cStride_tmp, &z_hStride_tmp, &z_wStride_tmp);
        size_t z_nStride = (size_t)z_c * (size_t)z_h * (size_t)z_w, z_cStride = (size_t)z_h * (size_t)z_w, z_hStride = (size_t)z_w, z_wStride = 1;

        int pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w;
        cudnnConvolutionMode_t mode;
        cudnnDataType_t computeType;
        cudnnGetConvolution2dDescriptor(convDesc, &pad_h, &pad_w, &stride_h, &stride_w, &dilation_h, &dilation_w, &mode, &computeType);

        int dilation_w_h = (w_h - 1) * dilation_h + 1;

        cudnnPoolingMode_t pool_mode;
        cudnnNanPropagation_t maxpoolingNanOpt;
        int pool_window_h, pool_window_w, pool_pad_h, pool_pad_w, pool_stride_h, pool_stride_w;
        cudnnGetPooling2dDescriptor(poolingDesc, &pool_mode, &maxpoolingNanOpt, &pool_window_h, &pool_window_w, &pool_pad_h, &pool_pad_w, &pool_stride_h, &pool_stride_w);


        size_t x_data_size = cudnnSizeOf(x_dataType);
        size_t w_data_size = cudnnSizeOf(w_dataType);
        size_t y_data_size = cudnnSizeOf(y_dataType);
        size_t bias_data_size = cudnnSizeOf(bias_dataType);
        size_t z_data_size = cudnnSizeOf(z_dataType);


        // check parameter 
        if((n != y_n) || (x_c != w_c) || (w_k != y_c) || (bias_n != 1) || (bias_c != y_c) || (bias_h != 1) || (bias_w != 1)){
                printf("ooc_cudnnConvolutionBiasActivationPoolingForward : CUDNN_STATUS_BAD_PARAM\n");
                printf("%d, %d\n", n, y_n);
                printf("%d, %d\n", x_c, w_c);
                printf("%d, %d\n", w_k, y_c);
                printf("%d, %d, %d, %d\n", bias_n, bias_c, bias_h, bias_w);
                return CUDNN_STATUS_BAD_PARAM;
        }
        if((x_format != CUDNN_TENSOR_NCHW) || (y_format != CUDNN_TENSOR_NCHW) || (pool_pad_h != 0) || (pool_window_h != pool_stride_h)){
                printf("ooc_cudnnConvolutionBiasActivationPoolingForward : CUDNN_STATUS_NOT_SUPPORTED\n");
                return CUDNN_STATUS_NOT_SUPPORTED;
        }

        
        // optimize
        int n_step, x_c_step, y_c_step, h_step;
        int num_pipeline = 1;

        ooc_cudnnConvolutionBiasActivationPoolingForward_optimize(
                            n, x_c, x_h, x_w, w_h, w_w, y_c, y_h, y_w, z_h, z_w, x_on_device, w_on_device, bias_on_device, y_on_device, z_on_device,
                            stride_h, pool_window_h, pool_window_w, pool_stride_h, dilation_h, y_data_size, &n_step, &y_c_step, &x_c_step, &h_step);
        


        if(n_step * y_c_step * h_step > 1){
            if(n_step * y_c_step * h_step == 2) num_pipeline = 2;
            else num_pipeline = 3;
        }

        int d_n = n / n_step; if(0 < n % n_step) d_n++;
        int d_y_c = y_c / y_c_step; if(0 < y_c % y_c_step) d_y_c++;
        int d_x_c = x_c / x_c_step; if(0 < x_c % x_c_step) d_x_c++;
        int d_y_h = y_h / h_step; if(0 < y_h % h_step) d_y_h++;
        int d_x_h = (d_y_h - 1) * stride_h + dilation_w_h;
        int d_z_h = (d_y_h + 2 * pool_pad_h - pool_window_h) / pool_stride_h + 1;
        

        // initialize d_x
        void *d_x[3], *d_x_buffer[3];
        size_t d_x_size = d_n * d_x_c * d_x_h * x_w * x_data_size;
        if(!x_on_device || (pad_h != 0 && h_step != 1)){
                for(i = 0; i < num_pipeline; i++){
                        cudaMalloc((void**)&d_x[i], d_x_size);
                        d_x_buffer[i] = d_x[i];
                }
        }
        cudnnTensorDescriptor_t d_xDesc;
        cudnnCreateTensorDescriptor(&d_xDesc);

        // initialize d_w
        void *d_w[3];
        size_t d_w_size = d_y_c * d_x_c * w_h * w_w * w_data_size;
        if((!w_on_device) || (d_x_c != x_c)){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_w[i], d_w_size);
        }
        cudnnFilterDescriptor_t d_wDesc;
        cudnnCreateFilterDescriptor(&d_wDesc);


        // initialize d_y
        void *d_y[3];
        size_t d_y_size = d_n * d_y_c * d_y_h * y_w * y_data_size;
        if(!y_on_device){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_y[i], d_y_size);
        }
        cudnnTensorDescriptor_t d_yDesc;
        cudnnCreateTensorDescriptor(&d_yDesc);


        // initialize d_bias
        void *d_bias[3];
        size_t d_bias_size = 1 * d_y_c * 1 * 1 * bias_data_size;
        if(!bias_on_device){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_bias[i], d_bias_size);
        }
        cudnnTensorDescriptor_t d_biasDesc;
        cudnnCreateTensorDescriptor(&d_biasDesc);


        // initialize d_z
        void *d_z[3];
        size_t d_z_size = d_n * d_y_c * d_z_h * z_w * z_data_size;
        if(!z_on_device){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_z[i], d_z_size);
        }
        cudnnTensorDescriptor_t d_zDesc;
        cudnnCreateTensorDescriptor(&d_zDesc);


        // initialize d_convDesc
        cudnnConvolutionDescriptor_t d_convDesc;
        cudnnCreateConvolutionDescriptor(&d_convDesc);

        // initialize d_poolingDesc
        cudnnPoolingDescriptor_t d_poolingDesc;
        cudnnCreatePoolingDescriptor(&d_poolingDesc);


        // convolution loop
        int i_n = 0, i_x_c = 0, i_y_c = 0, i_h;

        int n_count, x_c_count, y_c_count, x_h_count, y_h_count, z_h_count;

        int d_pad_h = 0;
        int d_pool_pad_h = 0;

        float one = 1.0f;  // ToDo : floatでいいのか
        float zero = 0.0f;



        // initialize n_count
        n_count = 0;

        for(i_n = 0; i_n < n_step; i_n++){

                d_n = n / n_step;
                if(i_n < n % n_step) d_n++;

                // initialize y_c_count
                y_c_count = 0;

                for(i_y_c = 0; i_y_c < y_c_step; i_y_c++){

                        d_y_c = y_c / y_c_step;
                        if(i_y_c < y_c % y_c_step) d_y_c++;

                        // initialize y_h_count
                        y_h_count = 0;
                        x_h_count = y_h_count * stride_h - pad_h;

                        for(i_h = 0; i_h < h_step; i_h++){

                                d_y_h = y_h / h_step;
                                if(i_h < y_h % h_step) d_y_h++;

                                d_x_h = (d_y_h - 1) * stride_h + dilation_w_h;

                                if(h_step == 1){ d_y_h = y_h; x_h_count = 0; d_x_h = x_h; d_pad_h = pad_h; }

                                // update d_y
                                if(y_on_device){
                                        cudnnSetTensor4dDescriptorEx(d_yDesc, y_dataType, d_n, d_y_c, d_y_h, y_w, y_nStride, y_cStride, y_hStride, y_wStride);
                                        d_y[current] = y + (n_count * y_nStride + y_c_count * y_cStride + y_h_count * y_hStride) * y_data_size;
                                }
                                else{
                                        cudnnSetTensor4dDescriptor(d_yDesc, y_format, y_dataType, d_n, d_y_c, d_y_h, y_w);
                                }

    
                                // update d_convDesc
                                cudnnSetConvolution2dDescriptor(d_convDesc, d_pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, mode, computeType);


                                // initialize x_c_count
                                x_c_count = 0;

                                for(i_x_c = 0; i_x_c < x_c_step; i_x_c++){

                                        d_x_c = x_c / x_c_step;
                                        if(i_x_c < x_c % x_c_step) d_x_c++;

                                        // update d_x
                                        if(x_on_device && (pad_h == 0 || h_step == 1 || (i_h != 0 && i_h != h_step-1))){
                                                cudnnSetTensor4dDescriptorEx(d_xDesc, x_dataType, d_n, d_x_c, d_x_h, x_w, x_nStride, x_cStride, x_hStride, x_wStride);
                                                d_x[current] = (void *)x + (n_count * x_nStride + x_c_count * x_cStride + x_h_count * x_hStride) * x_data_size;
                                        }
                                        else{
                                                d_x[current] = d_x_buffer[current];
                                                cudnnSetTensor4dDescriptor(d_xDesc, x_format, x_dataType, d_n, d_x_c, d_x_h, x_w);
                                                ooc_cudnnGetSubTensor4D(handle, xDesc, x, d_xDesc, d_x[current], n_count, x_c_count, x_h_count, 0);
                                        }

                                        // update d_w
                                        cudnnSetFilter4dDescriptor(d_wDesc, w_dataType, w_format, d_y_c, d_x_c, w_h, w_w);
                                        if(w_on_device && (d_x_c == x_c)){
                                                d_w[current] = (void *)w + y_c_count * x_c * w_h * w_w * w_data_size; 
                                        }
                                        else{
                                                ooc_cudnnGetSubFilter4D(handle, wDesc, w, d_wDesc, d_w[current], y_c_count, x_c_count, 0, 0);
                                        }

                                        cudaStreamWaitEvent(streams[current], flags[current], 0);

                                        // convolution main
                                        if(i_x_c == 0){
                                                status = cudnnConvolutionForward(handle, &one, d_xDesc, d_x[current], d_wDesc, d_w[current], 
                                                                d_convDesc, algo, workSpace, workSpaceSizeInBytes, &zero, d_yDesc, d_y[current]);
                                        }
                                        else{
                                                status = cudnnConvolutionForward(handle, &one, d_xDesc, d_x[current], d_wDesc, d_w[current], 
                                                                d_convDesc, algo, workSpace, workSpaceSizeInBytes, &one, d_yDesc, d_y[current]);
                                        }



                                        x_c_count += d_x_c;
                                }


                                cudaEventRecord(flags[current], streams[current]);
                                cudnnSetStream(handle, streams_D2H[current]);
                                cudaStreamWaitEvent(streams_D2H[current], flags[current], 0);
                                cudaStreamWaitEvent(streams_D2H[current], flags[(current + num_pipeline - 1) % num_pipeline], 0);


                                // update d_bias
                                if(bias_on_device){
                                        cudnnSetTensor4dDescriptor(d_biasDesc, CUDNN_TENSOR_NCHW, bias_dataType, 1, d_y_c, 1, 1);
                                        d_bias[current] = (void *)bias + (y_c_count * bias_cStride) * bias_data_size;
                                }
                                else{
                                        cudnnSetTensor4dDescriptor(d_biasDesc, bias_format, bias_dataType, 1, d_y_c, 1, 1);
                                        ooc_cudnnGetSubTensor4D(handle, biasDesc, bias, d_biasDesc, d_bias[current], 0, y_c_count, 0, 0);
                                }


                                //add bias
                                status = cudnnAddTensor(handle, &one, d_biasDesc, d_bias[current], &one, d_yDesc, d_y[current]);

                                // activation
                                status = cudnnActivationForward(handle, activationDesc, &one, d_yDesc, d_y[current], &zero, d_yDesc, d_y[current]);


                                // update h_y
                                if(!y_on_device){
                                        ooc_cudnnSetSubTensor4D(handle, yDesc, y, d_yDesc, d_y[current], n_count, y_c_count, y_h_count, 0);
                                }


                                // pooling
                                z_h_count = (y_h_count + pool_pad_h) / pool_stride_h;
                                d_z_h = (d_y_h + 2 * pool_pad_h - pool_window_h) / pool_stride_h + 1;
                                if(z_on_device){
                                        cudnnSetTensor4dDescriptorEx(d_zDesc, y_dataType, d_n, d_y_c, d_z_h, z_w, z_nStride, z_cStride, z_hStride, z_wStride);
                                        d_z[current] = z + (n_count * z_nStride + y_c_count * z_cStride + z_h_count * z_hStride) * y_data_size;
                                }
                                else cudnnSetTensor4dDescriptor(d_zDesc, y_format, y_dataType, d_n, d_y_c, d_z_h, z_w);
                                cudnnSetPooling2dDescriptor(d_poolingDesc, pool_mode, maxpoolingNanOpt, pool_window_h, pool_window_w, d_pool_pad_h, pool_pad_w, pool_stride_h, pool_stride_w);

                                status = cudnnPoolingForward(handle, d_poolingDesc, &one, d_yDesc, d_y[current], &zero, d_zDesc, d_z[current]);
                                if(!z_on_device) ooc_cudnnSetSubTensor4D(handle, zDesc, z, d_zDesc, d_z[current], n_count, y_c_count, z_h_count, 0);


                                cudaEventRecord(flags[current], streams_D2H[current]);
                                
                                y_h_count += d_y_h;
                                x_h_count = y_h_count * stride_h - pad_h;


                                current = (current + 1) % num_pipeline;
                                cudnnSetStream(handle, streams[current]);
                        }


                        y_c_count += d_y_c;
                }


                n_count += d_n;
        }
        

        // finalize stream 
        cudnnSetStream(handle, stream_handle);
        for(i = 0; i < 3; i++){
                cudaStreamSynchronize(streams[i]);
                cudaStreamSynchronize(streams_D2H[i]);
        }

        for(i = 0; i < 3; i++) cudaEventDestroy(flags[i]);


        cudnnDestroyTensorDescriptor(d_xDesc);
        cudnnDestroyFilterDescriptor(d_wDesc);
        cudnnDestroyTensorDescriptor(d_yDesc);
        cudnnDestroyTensorDescriptor(d_zDesc);
        cudnnDestroyTensorDescriptor(d_biasDesc);
        cudnnDestroyConvolutionDescriptor(d_convDesc);


        if(!x_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_x[i]);
        if((!w_on_device) || (d_x_c != x_c)) for(i = 0; i < num_pipeline; i++) cudaFree(d_w[i]);
        if(!y_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_y[i]);
        if(!z_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_z[i]);
        if(!bias_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_bias[i]);

        
        return status;
}