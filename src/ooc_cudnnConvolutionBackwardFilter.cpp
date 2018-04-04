
#include "ooc_cudnn.h"

#include <sys/time.h>


cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBackwardFilter_optimize(
                                const size_t                        n,
                                const size_t                        x_c,
                                const size_t                        x_h,
                                const size_t                        x_w,
                                const size_t                        w_h,
                                const size_t                        w_w,
                                const size_t                        y_c,
                                const size_t                        y_h,
                                const size_t                        y_w,
                                const bool                          x_on_device,
                                const bool                          dw_on_device,
                                const bool                          dy_on_device,
                                const size_t                        stride_h,
                                const size_t                        dilation_h,
                                const size_t                        data_size,
                                int                                *n_step,
                                int                                *x_c_step,
                                int                                *y_c_step,
                                int                                *h_step );




cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBackwardFilter(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionBwdFilterAlgo_t     algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const cudnnFilterDescriptor_t       dwDesc,
                                void                               *dw ){

        cudnnStatus_t status;
        int i, j, tmp1, tmp2;


        cudaPointerAttributes x_attr;
        cudaPointerAttributes dw_attr;
        cudaPointerAttributes dy_attr;

        cudaPointerGetAttributes(&x_attr, x);
        cudaPointerGetAttributes(&dw_attr, dw);
        cudaPointerGetAttributes(&dy_attr, dy);

        bool x_on_device = (x_attr.memoryType == cudaMemoryTypeDevice);
        bool dw_on_device = (dw_attr.memoryType == cudaMemoryTypeDevice);
        bool dy_on_device = (dy_attr.memoryType == cudaMemoryTypeDevice);


        
        // initialize stream 
        cudaStream_t stream_handle;
        cudnnGetStream(handle, &stream_handle);

        int current = 0; 

        cudaStream_t streams[3];
        streams[0] = stream_handle;
        for(i = 1; i < 3; i++) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

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

        cudnnDataType_t dw_dataType;
        cudnnTensorFormat_t dw_format;
        int w_k, w_c, w_h, w_w;
        cudnnGetFilter4dDescriptor(dwDesc, &dw_dataType, &dw_format, &w_k, &w_c, &w_h, &w_w);

        cudnnDataType_t dy_dataType;
        cudnnTensorFormat_t dy_format = CUDNN_TENSOR_NCHW;
        int y_n, y_c, y_h, y_w, y_nStride_tmp, y_cStride_tmp, y_hStride_tmp, y_wStride_tmp;
        cudnnGetTensor4dDescriptor(dyDesc, &dy_dataType, &y_n, &y_c, &y_h, &y_w, &y_nStride_tmp, &y_cStride_tmp, &y_hStride_tmp, &y_wStride_tmp);
        size_t y_nStride = (size_t)y_c * (size_t)y_h * (size_t)y_w, y_cStride = (size_t)y_h * (size_t)y_w, y_hStride = (size_t)y_w, y_wStride = 1;


        int pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w;
        cudnnConvolutionMode_t mode;
        cudnnDataType_t computeType;
        cudnnGetConvolution2dDescriptor(convDesc, &pad_h, &pad_w, &stride_h, &stride_w, &dilation_h, &dilation_w, &mode, &computeType);

        int dilation_w_h = (w_h - 1) * dilation_h + 1;

        size_t x_data_size = cudnnSizeOf(x_dataType);
        size_t dw_data_size = cudnnSizeOf(dw_dataType);
        size_t dy_data_size = cudnnSizeOf(dy_dataType);


        // check parameter 
        if((n != y_n) || (x_c != w_c) || (w_k != y_c)){
                printf("ooc_cudnnConvolutionBackwardFilter : CUDNN_STATUS_BAD_PARAM\n");
                printf("%d, %d\n", n, y_n);
                printf("%d, %d\n", x_c, w_c);
                printf("%d, %d\n", w_k, y_c);
                return CUDNN_STATUS_BAD_PARAM;
        }
        if((x_format != CUDNN_TENSOR_NCHW) || (dy_format != CUDNN_TENSOR_NCHW)){
                printf("ooc_cudnnConvolutionBackwardFilter : CUDNN_STATUS_NOT_SUPPORTED\n");
                return CUDNN_STATUS_NOT_SUPPORTED;
        }

        
        // optimize
        int n_step, x_c_step, y_c_step, h_step;
        int num_pipeline = 1;

        ooc_cudnnConvolutionBackwardFilter_optimize(
                            n, x_c, x_h, x_w, w_h, w_w, y_c, y_h, y_w, x_on_device, dw_on_device, dy_on_device,
                            stride_h, dilation_h, x_data_size, &n_step, &x_c_step, &y_c_step, &h_step);


        if(n_step * x_c_step * h_step > 1){
            if(n_step * x_c_step * h_step == 2) num_pipeline = 2;
            else num_pipeline = 3;
        }

        int d_n = n / n_step; if(0 < n % n_step) d_n++;
        int d_x_c = x_c / x_c_step; if(0 < x_c % x_c_step) d_x_c++;
        int d_y_c = y_c / y_c_step; if(0 < y_c % y_c_step) d_y_c++;
        int d_x_h = x_h / h_step + stride_h; if(0 < x_h % h_step) d_x_h++;
        int d_y_h = d_x_h - 1 + dilation_w_h;


        // initialize d_x
        void *d_x[3];
        size_t d_x_size = d_n * d_x_c * d_x_h * x_w * x_data_size;

        if(!x_on_device){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_x[i], d_x_size);
        }

        cudnnTensorDescriptor_t d_xDesc;
        cudnnCreateTensorDescriptor(&d_xDesc);

        // initialize d_dw
        void *d_dw[3];
        size_t d_dw_size = d_y_c * d_x_c * w_h * w_w * dw_data_size;

        if((!dw_on_device) || (d_x_c != x_c)){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_dw[i], d_dw_size);
        }

        cudnnFilterDescriptor_t d_dwDesc;
        cudnnCreateFilterDescriptor(&d_dwDesc);


        // initialize d_dy
        void *d_dy[3];
        size_t d_dy_size = d_n * d_y_c * d_y_h * y_w * dy_data_size;

        if(!dy_on_device){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_dy[i], d_dy_size);
        }

        cudnnTensorDescriptor_t d_dyDesc;
        cudnnCreateTensorDescriptor(&d_dyDesc);

        // initialize d_convDesc
        cudnnConvolutionDescriptor_t d_convDesc;
        cudnnCreateConvolutionDescriptor(&d_convDesc);


        // convolution loop
        int i_n, i_x_c, i_y_c, i_h;

        int n_count, x_c_count, y_c_count, x_h_count, y_h_count;

        int pad_h_top = pad_h;
        if(x_h != (y_h - 1) * stride_h + dilation_w_h - 2 * pad_h) pad_h_top--;

        int d_pad_h;

        float one = 1.0f;
        float zero = 0.0f;
        float *beta_ptr[3];



        // initialize n_count
        n_count = 0;

        for(i_n = 0; i_n < n_step; i_n++){

                d_n = n / n_step;
                if(i_n < n % n_step) d_n++;


                // initialize x_c_count
                x_c_count = 0;

                for(i_x_c = 0; i_x_c < x_c_step; i_x_c++){

                        d_x_c = x_c / x_c_step;
                        if(i_x_c < x_c % x_c_step) d_x_c++;

                        // initialize x_h_count
                        x_h_count = 0;

                        tmp1 = x_h_count - (dilation_w_h - 1 - pad_h_top);
                        while(tmp1 < 0) tmp1 += stride_h;  // ensure positive.
                        tmp2 = (stride_h - tmp1 % stride_h) % stride_h;
                        d_pad_h = dilation_w_h - 1 - tmp2;

                        tmp1 = x_h_count - (dilation_w_h - 1 - pad_h_top);
                        y_h_count = (tmp1 + tmp2) / stride_h;


                        for(i_h = 0; i_h < h_step; i_h++){

                                tmp1 = stride_h - (2 * d_pad_h - dilation_w_h);
                                while(tmp1 < 0) tmp1 += stride_h;  // ensure positive.
                                tmp1 %= stride_h;

                                tmp2 = ((x_h - x_h_count - tmp1) / stride_h) / (h_step - i_h);
                                if(tmp2 <= 0) tmp2 = 1;
                                d_x_h = tmp2 * stride_h + tmp1;


                                d_y_h = (d_x_h + (2 * d_pad_h - dilation_w_h)) / stride_h + 1;

                                if(h_step == 1){ d_x_h = x_h; y_h_count = 0; d_y_h = y_h; d_pad_h = pad_h; }


                                // update d_x
                                if(x_on_device){
                                        cudnnSetTensor4dDescriptorEx(d_xDesc, x_dataType, d_n, d_x_c, d_x_h, x_w, x_nStride, x_cStride, x_hStride, x_wStride);
                                        d_x[current] = (void *)x + (n_count * x_nStride + x_c_count * x_cStride + x_h_count * x_hStride) * x_data_size;
                                }
                                else{
                                        cudnnSetTensor4dDescriptor(d_xDesc, x_format, x_dataType, d_n, d_x_c, d_x_h, x_w);
                                        ooc_cudnnGetSubTensor4D(handle, xDesc, x, d_xDesc, d_x[current], n_count, x_c_count, x_h_count, 0);
                                }
    
                                // update d_convDesc
                                cudnnSetConvolution2dDescriptor(d_convDesc, d_pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, mode, computeType);


                                // initialize y_c_count
                                y_c_count = 0;

                                for(i_y_c = 0; i_y_c < y_c_step; i_y_c++){

                                        d_y_c = y_c / y_c_step;
                                        if(i_y_c < y_c % y_c_step) d_y_c++;

                                        // update d_dy
                                        if(dy_on_device){
                                                cudnnSetTensor4dDescriptorEx(d_dyDesc, dy_dataType, d_n, d_y_c, d_y_h, y_w, y_nStride, y_cStride, y_hStride, y_wStride);
                                                d_dy[current] = (void *)dy + (n_count * y_nStride + y_c_count * y_cStride + y_h_count * y_hStride) * dy_data_size;
                                        }
                                        else{
                                                cudnnSetTensor4dDescriptor(d_dyDesc, dy_format, dy_dataType, d_n, d_y_c, d_y_h, y_w);
                                                ooc_cudnnGetSubTensor4D(handle, dyDesc, dy, d_dyDesc, d_dy[current], n_count, y_c_count, y_h_count, 0);
                                        }

                                        // update d_dw
                                        cudaStreamWaitEvent(streams[current], flags[(current + num_pipeline - 1) % num_pipeline], 0);
                                        cudnnSetFilter4dDescriptor(d_dwDesc, dw_dataType, dw_format, d_y_c, d_x_c, w_h, w_w);
                                        if(dw_on_device && (d_x_c == x_c)){
                                                d_dw[current] = dw + y_c_count * x_c * w_h * w_w * dw_data_size; 
                                        }
                                        else{
                                                ooc_cudnnGetSubFilter4D(handle, dwDesc, dw, d_dwDesc, d_dw[current], y_c_count, x_c_count, 0, 0);
                                        }


                                        // convolution main
                                        if(i_n == 0 && i_h == 0) beta_ptr[current] = &zero;
                                        else beta_ptr[current] = &one;
                                        status = cudnnConvolutionBackwardFilter(handle, alpha, d_xDesc, d_x[current], d_dyDesc, d_dy[current], 
                                                        d_convDesc, algo, workSpace, workSpaceSizeInBytes, beta_ptr[current], d_dwDesc, d_dw[current]);


                                        // update h_dw
                                        if(!(dw_on_device && (d_x_c == x_c))){
                                                ooc_cudnnSetSubFilter4D(handle, dwDesc, dw, d_dwDesc, d_dw[current], y_c_count, x_c_count, 0, 0);
                                        }
                                        cudaEventRecord(flags[current], streams[current]);


                                        y_c_count += d_y_c;
                                }
                                
                                x_h_count += d_x_h;
                                if(x_h_count >= x_h) i_h = h_step;

                                tmp1 = x_h_count - (dilation_w_h - 1 - pad_h_top);
                                while(tmp1 < 0) tmp1 += stride_h;  // ensure positive.
                                tmp2 = (stride_h - tmp1 % stride_h) % stride_h;
                                d_pad_h = dilation_w_h - 1 - tmp2;

                                tmp1 = x_h_count - (dilation_w_h - 1 - pad_h_top);
                                if(tmp1 < 0) tmp1 += tmp2;  // ensure truncation.
                                y_h_count = (tmp1 + tmp2) / stride_h;


                                current = (current + 1) % num_pipeline;
                                cudnnSetStream(handle, streams[current]);
                        }


                        x_c_count += d_x_c;
                }


                n_count += d_n;
        }
        

        // finalize stream 
        cudnnSetStream(handle, stream_handle);
        for(i = 0; i < 3; i++){
                cudaStreamSynchronize(streams[i]);
        }

        for(i = 0; i < 3; i++) cudaEventDestroy(flags[i]);


        cudnnDestroyTensorDescriptor(d_xDesc);
        cudnnDestroyFilterDescriptor(d_dwDesc);
        cudnnDestroyTensorDescriptor(d_dyDesc);
        cudnnDestroyConvolutionDescriptor(d_convDesc);


        if(!x_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_x[i]);
        if((!dw_on_device) || (d_x_c != x_c)) for(i = 0; i < num_pipeline; i++) cudaFree(d_dw[i]);
        if(!dy_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_dy[i]);

        
        return status;
}
