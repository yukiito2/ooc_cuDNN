
#include "ooc_cudnn.h"

#include <sys/time.h>


cudnnStatus_t CUDNNWINAPI ooc_cudnnActivationConvolutionBackwardDataFilterBias_optimize(
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
                                const bool                          w_on_device,
                                const bool                          y_on_device,
                                const bool                          dy_on_device,
                                const bool                          dx_on_device,
                                const bool                          dw_on_device,
                                const bool                          db_on_device,
                                const size_t                        stride_h,
                                const size_t                        dilation_h,
                                const size_t                        data_size,
                                int                                *n_step,
                                int                                *x_c_step,
                                int                                *y_c_step,
                                int                                *h_step );



//y_cを分割するとBackwardFilter関連の同期によりパイプラインがうまくいかないため、現在の実装ではy_c_step=1で固定
cudnnStatus_t CUDNNWINAPI ooc_cudnnActivationConvolutionBackwardDataFilterBias(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnActivationDescriptor_t   activationDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionBwdDataAlgo_t       algo_data,
                                void                               *workSpace_data,
                                size_t                              workSpaceSizeInBytes_data,
                                cudnnConvolutionBwdFilterAlgo_t     algo_filter,
                                void                               *workSpace_filter,
                                size_t                              workSpaceSizeInBytes_filter,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx,
                                const cudnnFilterDescriptor_t       dwDesc,
                                void                               *dw,
                                const cudnnTensorDescriptor_t       dbDesc,
                                void                               *db ){



        cudnnStatus_t status;

        cudaPointerAttributes x_attr;
        cudaPointerAttributes w_attr;
        cudaPointerAttributes y_attr;
        cudaPointerAttributes dy_attr;
        cudaPointerAttributes dx_attr;
        cudaPointerAttributes dw_attr;
        cudaPointerAttributes db_attr;

        cudaPointerGetAttributes(&x_attr, x);
        cudaPointerGetAttributes(&w_attr, w);
        cudaPointerGetAttributes(&y_attr, y);
        cudaPointerGetAttributes(&dy_attr, dy);
        cudaPointerGetAttributes(&dx_attr, dx);
        cudaPointerGetAttributes(&dw_attr, dw);
        cudaPointerGetAttributes(&db_attr, db);

        bool x_on_device = (x_attr.memoryType == cudaMemoryTypeDevice);
        bool w_on_device = (w_attr.memoryType == cudaMemoryTypeDevice);
        bool y_on_device = (y_attr.memoryType == cudaMemoryTypeDevice);
        bool dy_on_device = (dy_attr.memoryType == cudaMemoryTypeDevice);
        bool dx_on_device = (dx_attr.memoryType == cudaMemoryTypeDevice);
        bool dw_on_device = (dw_attr.memoryType == cudaMemoryTypeDevice);
        bool db_on_device = (db_attr.memoryType == cudaMemoryTypeDevice);


        int i, j, tmp1, tmp2;

        // initialize stream 
        cudaStream_t stream_handle;
        cudnnGetStream(handle, &stream_handle);

        int current = 0; 

        int low_priority, high_priority;
        cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority);

        cudaStream_t streams[3];
        streams[0] = stream_handle;
        for(i = 1; i < 3; i++) cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, low_priority);

        cudaStream_t streams_high[3];
        for(i = 0; i < 3; i++) cudaStreamCreateWithPriority(&streams_high[i], cudaStreamNonBlocking, high_priority);

        cudaEvent_t flags[3];
        for(i = 0; i < 3; i++){
                cudaEventCreate(&flags[i]);
                cudaEventRecord(flags[i], streams[i]);
        }

        cudaEvent_t flags_backfilter[3];
        for(i = 0; i < 3; i++){
                cudaEventCreate(&flags_backfilter[i]);
                cudaEventRecord(flags_backfilter[i], streams[i]);
        }

        cudaEvent_t flags_backbias[3];
        for(i = 0; i < 3; i++){
                cudaEventCreate(&flags_backbias[i]);
                cudaEventRecord(flags_backbias[i], streams[i]);
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

        cudnnDataType_t dy_dataType;
        cudnnTensorFormat_t dy_format = CUDNN_TENSOR_NCHW;
        int dy_n, dy_c, dy_h, dy_w, dy_nStride_tmp, dy_cStride_tmp, dy_hStride_tmp, dy_wStride_tmp;
        cudnnGetTensor4dDescriptor(dyDesc, &dy_dataType, &dy_n, &dy_c, &dy_h, &dy_w, &dy_nStride_tmp, &dy_cStride_tmp, &dy_hStride_tmp, &dy_wStride_tmp);
        size_t dy_nStride = (size_t)dy_c * (size_t)dy_h * (size_t)dy_w, dy_cStride = (size_t)dy_h * (size_t)dy_w, dy_hStride = (size_t)dy_w, dy_wStride = 1;

        cudnnDataType_t db_dataType;
        cudnnTensorFormat_t db_format = CUDNN_TENSOR_NCHW;
        int b_n, b_c, b_h, b_w, b_nStride_tmp, b_cStride_tmp, b_hStride_tmp, b_wStride_tmp;
        cudnnGetTensor4dDescriptor(dbDesc, &db_dataType, &b_n, &b_c, &b_h, &b_w, &b_nStride_tmp, &b_cStride_tmp, &b_hStride_tmp, &b_wStride_tmp);
        size_t b_nStride = (size_t)b_c * (size_t)b_h * (size_t)b_w, b_cStride = (size_t)b_h * (size_t)b_w, b_hStride = (size_t)b_w, b_wStride = 1;

        int pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w;
        cudnnConvolutionMode_t mode;
        cudnnDataType_t computeType;
        cudnnGetConvolution2dDescriptor(convDesc, &pad_h, &pad_w, &stride_h, &stride_w, &dilation_h, &dilation_w, &mode, &computeType);

        int dilation_w_h = (w_h - 1) * dilation_h + 1;

        cudnnActivationMode_t active_mode;
        cudnnNanPropagation_t reluNanOpt;
        double reluCeiling;
        cudnnGetActivationDescriptor(activationDesc, &active_mode, &reluNanOpt, &reluCeiling);


        size_t x_data_size = cudnnSizeOf(x_dataType);
        size_t w_data_size = cudnnSizeOf(w_dataType);
        size_t y_data_size = cudnnSizeOf(dy_dataType);
        size_t db_data_size = cudnnSizeOf(db_dataType);


        // check parameter 
        if((n != dy_n) || (w_k != y_c) || (x_c != w_c) || (b_n != 1) || (b_c != y_c) || (b_h != 1) || (b_w != 1)){
                printf("ooc_cudnnActivationConvolutionBackwardDataFilterBias : CUDNN_STATUS_BAD_PARAM\n");
                return CUDNN_STATUS_BAD_PARAM;
        }
        if((x_format != CUDNN_TENSOR_NCHW) || (dy_format != CUDNN_TENSOR_NCHW) || (active_mode == CUDNN_ACTIVATION_CLIPPED_RELU)){
                printf("ooc_cudnnActivationConvolutionBackwardDataFilterBias : CUDNN_STATUS_NOT_SUPPORTED\n");
                return CUDNN_STATUS_NOT_SUPPORTED;
        }

        
        // optimize
        int n_step, x_c_step, y_c_step, h_step;
        int num_pipeline = 1;

        ooc_cudnnActivationConvolutionBackwardDataFilterBias_optimize(
                            n, x_c, x_h, x_w, w_h, w_w, y_c, y_h, y_w,
                            x_on_device, w_on_device, y_on_device, dy_on_device, dx_on_device, dw_on_device, db_on_device,
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
        if(!x_on_device){ for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_x[i], d_x_size); }
        cudnnTensorDescriptor_t d_xDesc;
        cudnnCreateTensorDescriptor(&d_xDesc);

        // initialize d_w
        void *d_w[3];
        size_t d_w_size = d_y_c * d_x_c * w_h * w_w * w_data_size;
        if((!w_on_device) || (d_x_c != x_c)){ for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_w[i], d_w_size); }
        cudnnFilterDescriptor_t d_wDesc;
        cudnnCreateFilterDescriptor(&d_wDesc);

        // initialize d_y
        void *d_y[3], *d_y_buffer[3];
        size_t d_y_size = d_n * d_y_c * d_y_h * y_w * y_data_size;
        if(!y_on_device || h_step != 1){
                for(i = 0; i < num_pipeline; i++){
                        cudaMalloc((void**)&d_y[i], d_y_size);
                        d_y_buffer[i] = d_y[i];
                }
        }
        cudnnTensorDescriptor_t d_yDesc;
        cudnnCreateTensorDescriptor(&d_yDesc);

        // initialize d_dy
        void *d_dy[3], *d_dy_buffer[3];
        size_t d_dy_size = d_n * d_y_c * d_y_h * y_w * y_data_size;
        if(!dy_on_device || h_step != 1){
                for(i = 0; i < num_pipeline; i++){
                        cudaMalloc((void**)&d_dy[i], d_dy_size);
                        d_dy_buffer[i] = d_dy[i];
                }
        }
        cudnnTensorDescriptor_t d_dyDesc;
        cudnnCreateTensorDescriptor(&d_dyDesc);

        // initialize d_dx
        void *d_dx[3];
        size_t d_dx_size = d_n * d_x_c * d_x_h * x_w * x_data_size;
        if(!dx_on_device){ for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_dx[i], d_dx_size); }
        cudnnTensorDescriptor_t d_dxDesc;
        cudnnCreateTensorDescriptor(&d_dxDesc);

        // initialize d_dw
        void *d_dw[3];
        size_t d_dw_size = d_y_c * d_x_c * w_h * w_w * w_data_size;
        if((!dw_on_device) || (d_x_c != x_c)){ for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_dw[i], d_dw_size); }
        cudnnFilterDescriptor_t d_dwDesc;
        cudnnCreateFilterDescriptor(&d_dwDesc);

        // initialize d_db
        void *d_db[3];
        size_t d_db_size = 1 * d_y_c * 1 * 1 * db_data_size;
        if(!db_on_device){ for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_db[i], d_db_size); }
        cudnnTensorDescriptor_t d_dbDesc;
        cudnnCreateTensorDescriptor(&d_dbDesc);

        // initialize d_convDesc
        cudnnConvolutionDescriptor_t d_convDesc;
        cudnnCreateConvolutionDescriptor(&d_convDesc);


        // convolution loop
        int i_n, i_x_c, i_y_c, i_h;

        int n_count, x_c_count, y_c_count, x_h_count, y_h_count; 

        int pad_h_top = pad_h;
        if(x_h != (y_h - 1) * stride_h + dilation_w_h - 2 * pad_h) pad_h_top--;

        int d_pad_h;


        //for back bias
        void *d_dy_backbias[3];
        int d_y_h_backbias, y_h_count_backbias;
        cudnnTensorDescriptor_t d_dyDesc_backbias;
        cudnnCreateTensorDescriptor(&d_dyDesc_backbias);
        int d_dy_nStride, d_dy_cStride, d_dy_hStride, d_dy_wStride; 


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
                        y_h_count_backbias = 0;


                        for(i_h = 0; i_h < h_step; i_h++){

                                tmp1 = stride_h - (2 * d_pad_h - dilation_w_h);
                                while(tmp1 < 0) tmp1 += stride_h;  // ensure positive.
                                tmp1 %= stride_h;

                                tmp2 = ((x_h - x_h_count - tmp1) / stride_h) / (h_step - i_h);
                                if(tmp2 <= 0) tmp2 = 1;
                                d_x_h = tmp2 * stride_h + tmp1;


                                d_y_h = (d_x_h + (2 * d_pad_h - dilation_w_h)) / stride_h + 1;

                                if(h_step == 1){ d_x_h = x_h; y_h_count = 0; d_y_h = y_h; d_pad_h = pad_h; }

                                d_y_h_backbias = y_h_count + d_y_h - y_h_count_backbias;
                                tmp1 = y_h - (y_h_count + d_y_h);
                                if(tmp1 < 0) d_y_h_backbias += tmp1; 
                                if(d_y_h_backbias < 0) d_y_h_backbias = 0;


                                // update d_x
                                if(x_on_device){
                                        cudnnSetTensor4dDescriptorEx(d_xDesc, x_dataType, d_n, d_x_c, d_x_h, x_w, x_nStride, x_cStride, x_hStride, x_wStride);
                                        d_x[current] = (void *)x + (n_count * x_nStride + x_c_count * x_cStride + x_h_count * x_hStride) * x_data_size;
                                }
                                else{
                                        cudnnSetTensor4dDescriptor(d_xDesc, x_format, x_dataType, d_n, d_x_c, d_x_h, x_w);
                                        ooc_cudnnGetSubTensor4D(handle, xDesc, x, d_xDesc, d_x[current], n_count, x_c_count, x_h_count, 0);
                                }


                                // update d_dx
                                if(dx_on_device){
                                        cudnnSetTensor4dDescriptorEx(d_dxDesc, x_dataType, d_n, d_x_c, d_x_h, x_w, x_nStride, x_cStride, x_hStride, x_wStride);
                                        d_dx[current] = dx + (n_count * x_nStride + x_c_count * x_cStride + x_h_count * x_hStride) * x_data_size;
                                }
                                else{ cudnnSetTensor4dDescriptor(d_dxDesc, x_format, x_dataType, d_n, d_x_c, d_x_h, x_w); }

    
                                // update d_convDesc
                                cudnnSetConvolution2dDescriptor(d_convDesc, d_pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, mode, computeType);


                                // initialize y_c_count
                                y_c_count = 0;

                                for(i_y_c = 0; i_y_c < y_c_step; i_y_c++){

                                        d_y_c = y_c / y_c_step;
                                        if(i_y_c < y_c % y_c_step) d_y_c++;

                                        // update d_dy, d_dy_backbias
                                        if(dy_on_device && (h_step == 1 || (i_h != 0 && i_h != h_step-1))){
                                                cudnnSetTensor4dDescriptorEx(d_dyDesc, dy_dataType, d_n, d_y_c, d_y_h, y_w, y_nStride, y_cStride, y_hStride, y_wStride);
                                                d_dy[current] = (void *)dy + (n_count * y_nStride + y_c_count * y_cStride + y_h_count * y_hStride) * y_data_size;

                                                cudnnSetTensor4dDescriptorEx(d_dyDesc_backbias, dy_dataType, d_n, d_y_c, d_y_h_backbias, y_w, y_nStride, y_cStride, y_hStride, y_wStride);
                                                d_dy_backbias[current] = (void *)dy + (n_count * y_nStride + y_c_count * y_cStride + y_h_count_backbias * y_hStride) * y_data_size;
                                        }
                                        else{
                                                d_dy[current] = d_dy_buffer[current];
                                                cudnnSetTensor4dDescriptor(d_dyDesc, dy_format, dy_dataType, d_n, d_y_c, d_y_h, y_w);
                                                ooc_cudnnGetSubTensor4D(handle, dyDesc, dy, d_dyDesc, d_dy[current], n_count, y_c_count, y_h_count, 0);

                                                d_dy_nStride = d_y_c * d_y_h * y_w;
                                                d_dy_cStride = d_y_h * y_w;
                                                d_dy_hStride = y_w;
                                                d_dy_wStride = 1;
                                                cudnnSetTensor4dDescriptorEx(d_dyDesc_backbias, dy_dataType, d_n, d_y_c, d_y_h_backbias, y_w, d_dy_nStride, d_dy_cStride, d_dy_hStride, d_dy_wStride);
                                                d_dy_backbias[current] = d_dy[current] + (y_h_count_backbias - y_h_count) * d_dy_hStride * y_data_size;
                                        }


                                        // activation bacward
                                        if(!(dy_on_device && (h_step == 1 || (i_h != 0 && i_h != h_step-1)) && i_x_c != 0)){
                                                // update d_y
                                                if(y_on_device && (h_step == 1 || (i_h != 0 && i_h != h_step-1))){
                                                        cudnnSetTensor4dDescriptorEx(d_yDesc, y_dataType, d_n, d_y_c, d_y_h, y_w, y_nStride, y_cStride, y_hStride, y_wStride);
                                                        d_y[current] = (void *)y + (n_count * y_nStride + y_c_count * y_cStride + y_h_count * y_hStride) * y_data_size;
                                                }
                                                else{
                                                        d_y[current] = d_y_buffer[current];
                                                        cudnnSetTensor4dDescriptor(d_yDesc, y_format, y_dataType, d_n, d_y_c, d_y_h, y_w);
                                                        ooc_cudnnGetSubTensor4D(handle, yDesc, y, d_yDesc, d_y[current], n_count, y_c_count, y_h_count, 0);
                                                }

                                                cudaEventRecord(flags[current], streams[current]);
                                                cudnnSetStream(handle, streams_high[current]);
                                                cudaStreamWaitEvent(streams_high[current], flags[current], 0);

                                                // activation main
                                                cudnnActivationBackward(handle, activationDesc, &one, d_yDesc, d_y[current], d_dyDesc, d_dy[current], d_yDesc, d_y[current], &zero, d_dyDesc, d_dy[current]);
                                        
                                                cudaEventRecord(flags[current], streams_high[current]);
                                                cudnnSetStream(handle, streams[current]);
                                                cudaStreamWaitEvent(streams[current], flags[current], 0);
                                        }


                                        // update d_w
                                        cudnnSetFilter4dDescriptor(d_wDesc, w_dataType, w_format, d_y_c, d_x_c, w_h, w_w);
                                        if(w_on_device && (d_x_c == x_c)){
                                                d_w[current] = (void *)w + y_c_count * x_c * w_h * w_w * w_data_size; 
                                        }
                                        else{ ooc_cudnnGetSubFilter4D(handle, wDesc, w, d_wDesc, d_w[current], y_c_count, x_c_count, 0, 0); }


                                        // update d_dw
                                        cudaStreamWaitEvent(streams[current], flags_backfilter[(current + num_pipeline - 1) % num_pipeline], 0);
                                        cudnnSetFilter4dDescriptor(d_dwDesc, w_dataType, w_format, d_y_c, d_x_c, w_h, w_w);
                                        if(dw_on_device && (d_x_c == x_c)){
                                                d_dw[current] = dw + y_c_count * x_c * w_h * w_w * w_data_size; 
                                        }
                                        else{ ooc_cudnnGetSubFilter4D(handle, dwDesc, dw, d_dwDesc, d_dw[current], y_c_count, x_c_count, 0, 0); }


                                        // update d_db
                                        cudaStreamWaitEvent(streams[current], flags_backbias[(current + num_pipeline - 1) % num_pipeline], 0);
                                        cudnnSetTensor4dDescriptor(d_dbDesc, CUDNN_TENSOR_NCHW, db_dataType, 1, d_y_c, 1, 1);
                                        if(db_on_device){ d_db[current] = db + (y_c_count * b_cStride) * db_data_size; }
                                        else{ ooc_cudnnGetSubTensor4D(handle, dbDesc, db, d_dbDesc, d_db[current], 0, y_c_count, 0, 0); }


                                        cudaStreamWaitEvent(streams[current], flags[current], 0);


                                        // convolution main
                                        // backward data
                                        if(i_y_c == 0){ beta_ptr[current] = &zero; }
                                        else{ beta_ptr[current] = &one; }
                                        status = cudnnConvolutionBackwardData(handle, &one, d_wDesc, d_w[current], d_dyDesc, d_dy[current], 
                                                        d_convDesc, algo_data, workSpace_data, workSpaceSizeInBytes_data, beta_ptr[current], d_dxDesc, d_dx[current]);


                                        // backward filter
                                        if((i_n == 0) && (i_h == 0)){ beta_ptr[current] = &zero; }
                                        else{ beta_ptr[current] = &one; }
                                        status = cudnnConvolutionBackwardFilter(handle, &one, d_xDesc, d_x[current], d_dyDesc, d_dy[current], 
                                                        d_convDesc, algo_filter, workSpace_filter, workSpaceSizeInBytes_filter, beta_ptr[current], d_dwDesc, d_dw[current]);

                                        // update h_dw
                                        if(!(dw_on_device && (d_x_c == x_c))){ ooc_cudnnSetSubFilter4D(handle, wDesc, dw, d_wDesc, d_dw[current], y_c_count, x_c_count, 0, 0); }
                                        cudaEventRecord(flags_backfilter[current], streams[current]);


                                        // backward bias
                                        if(i_x_c == 0){

                                                cudaEventRecord(flags[current], streams[current]);
                                                cudnnSetStream(handle, streams_high[current]);
                                                cudaStreamWaitEvent(streams_high[current], flags[current], 0);
                                                
                                                if((i_n == 0) && (i_h == 0)){ beta_ptr[current] = &zero; }
                                                else{ beta_ptr[current] = &one; }
                                                status = cudnnConvolutionBackwardBias(handle, &one, d_dyDesc_backbias, d_dy_backbias[current], beta_ptr[current], d_dbDesc, d_db[current]);                                        

                                                // update h_db
                                                if(!db_on_device){ ooc_cudnnSetSubTensor4D(handle, dbDesc, db, d_dbDesc, d_db[current], 0, y_c_count, 0, 0); }

                                                cudaEventRecord(flags[current], streams_high[current]);
                                                cudnnSetStream(handle, streams[current]);
                                                cudaStreamWaitEvent(streams[current], flags[current], 0);
                                                cudaEventRecord(flags_backbias[current], streams[current]);
                                        }

                                        y_c_count += d_y_c;
                                }

                                cudaEventRecord(flags[current], streams[current]);
                                cudnnSetStream(handle, streams_high[current]);
                                cudaStreamWaitEvent(streams_high[current], flags[current], 0);
                                cudaStreamWaitEvent(streams_high[current], flags[(current + num_pipeline - 1) % num_pipeline], 0);

                                // update h_dx
                                if(!dx_on_device){ ooc_cudnnSetSubTensor4D(handle, dxDesc, dx, d_dxDesc, d_dx[current], n_count, x_c_count, x_h_count, 0); }

                                cudaEventRecord(flags[current], streams_high[current]);



                                x_h_count += d_x_h;
                                if(x_h_count >= x_h) i_h = h_step;

                                tmp1 = x_h_count - (dilation_w_h - 1 - pad_h_top);
                                while(tmp1 < 0) tmp1 += stride_h;  // ensure positive.
                                tmp2 = (stride_h - tmp1 % stride_h) % stride_h;
                                d_pad_h = dilation_w_h - 1 - tmp2;

                                tmp1 = x_h_count - (dilation_w_h - 1 - pad_h_top);
                                if(tmp1 < 0) tmp1 += tmp2;  // ensure truncation.
                                y_h_count = (tmp1 + tmp2) / stride_h;
                                y_h_count_backbias += d_y_h_backbias;


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
                cudaStreamSynchronize(streams_high[i]);
        }

        for(i = 0; i < 3; i++) cudaEventDestroy(flags[i]);


        cudnnDestroyTensorDescriptor(d_xDesc);
        cudnnDestroyFilterDescriptor(d_wDesc);
        cudnnDestroyTensorDescriptor(d_yDesc);
        cudnnDestroyTensorDescriptor(d_dyDesc);
        cudnnDestroyTensorDescriptor(d_dyDesc_backbias);
        cudnnDestroyTensorDescriptor(d_dxDesc);
        cudnnDestroyFilterDescriptor(d_dwDesc);
        cudnnDestroyTensorDescriptor(d_dbDesc);
        cudnnDestroyConvolutionDescriptor(d_convDesc);


        if(!x_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_x[i]);
        if((!w_on_device) || (d_x_c != x_c)) for(i = 0; i < num_pipeline; i++) cudaFree(d_w[i]);
        if(!y_on_device || h_step != 1) for(i = 0; i < num_pipeline; i++) cudaFree(d_y[i]);
        if(!dy_on_device || h_step != 1) for(i = 0; i < num_pipeline; i++) cudaFree(d_dy[i]);
        if(!dx_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_dx[i]);
        if((!dw_on_device) || (d_x_c != x_c)) for(i = 0; i < num_pipeline; i++) cudaFree(d_dw[i]);
        if(!db_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_db[i]);

        
        return status;
}