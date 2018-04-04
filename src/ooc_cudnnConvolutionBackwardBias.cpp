
#include "ooc_cudnn.h"


cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBackwardBias_optimize(
                                const size_t                        n,
                                const size_t                        c,
                                const size_t                        h,
                                const size_t                        w,
                                const bool                          db_on_device,
                                const bool                          dy_on_device, 
                                const size_t                        data_size,
                                int                                *n_step,
                                int                                *c_step,
                                int                                *h_step );



// 現在bias用
cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBackwardBias(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dbDesc,
                                void                               *db ){

        int i, j;
        cudnnStatus_t status;

        cudaPointerAttributes db_attr;
        cudaPointerAttributes dy_attr;

        cudaPointerGetAttributes(&db_attr, db);
        cudaPointerGetAttributes(&dy_attr, dy);

        bool db_on_device = (db_attr.memoryType == cudaMemoryTypeDevice);
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
        cudnnDataType_t db_dataType;
        cudnnTensorFormat_t db_format = CUDNN_TENSOR_NCHW;
        int b_n, b_c, b_h, b_w, b_nStride_tmp, b_cStride_tmp, b_hStride_tmp, b_wStride_tmp;
        cudnnGetTensor4dDescriptor(dbDesc, &db_dataType, &b_n, &b_c, &b_h, &b_w, &b_nStride_tmp, &b_cStride_tmp, &b_hStride_tmp, &b_wStride_tmp);
        size_t b_nStride = (size_t)b_c * (size_t)b_h * (size_t)b_w, b_cStride = (size_t)b_h * (size_t)b_w, b_hStride = (size_t)b_w, b_wStride = 1;
    
        cudnnDataType_t dy_dataType;
        cudnnTensorFormat_t dy_format = CUDNN_TENSOR_NCHW;
        int n, y_c, y_h, y_w, nStride_tmp, y_cStride_tmp, y_hStride_tmp, y_wStride_tmp;
        cudnnGetTensor4dDescriptor(dyDesc, &dy_dataType, &n, &y_c, &y_h, &y_w, &nStride_tmp, &y_cStride_tmp, &y_hStride_tmp, &y_wStride_tmp);
        size_t nStride = (size_t)y_c * (size_t)y_h * (size_t)y_w, y_cStride = (size_t)y_h * (size_t)y_w, y_hStride = (size_t)y_w, y_wStride = 1;

        size_t dy_data_size = cudnnSizeOf(dy_dataType);
        size_t db_data_size = cudnnSizeOf(db_dataType);


        // check parameter 
        if((b_n != 1) || (b_c != y_c) || (b_h != 1) || (b_w != 1)){
                printf("ooc_cudnnConvolutionBackwardBias : CUDNN_STATUS_BAD_PARAM\n");
                printf("%d, %d\n", b_n, n);
                printf("%d, %d\n", b_c, y_c);
                printf("%d, %d\n", b_h, y_h);
                printf("%d, %d\n", b_w, y_w);
                return CUDNN_STATUS_BAD_PARAM;
        }
        if((dy_format != CUDNN_TENSOR_NCHW) || (db_format != CUDNN_TENSOR_NCHW)){
                printf("ooc_cudnnConvolutionBackwardBias : CUDNN_STATUS_NOT_SUPPORTED\n");
                return CUDNN_STATUS_NOT_SUPPORTED;
        }

        
        // optimize
        int n_step, c_step, h_step;
        int num_pipeline = 1;

        ooc_cudnnConvolutionBackwardBias_optimize(n, y_c, y_h, y_w,  db_on_device, dy_on_device, dy_data_size, &n_step, &c_step, &h_step);


        if(n_step * c_step * h_step > 1){
            if(n_step * c_step * h_step == 2) num_pipeline = 2;
            else num_pipeline = 3;
        }

        int d_n = n / n_step; if(0 < n % n_step) d_n++;
        int d_c = y_c / c_step; if(0 < y_c % c_step) d_c++;
        int d_h = y_h / h_step; if(0 < y_h % h_step) d_h++;
        

        // initialize d_db
        void *d_db[3];
        size_t d_db_size = 1 * d_c * 1 * 1 * db_data_size;

        if(!db_on_device){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_db[i], d_db_size);
        }

        cudnnTensorDescriptor_t d_dbDesc;
        cudnnCreateTensorDescriptor(&d_dbDesc);


        // initialize d_dy
        void *d_dy[3];
        size_t d_dy_size = d_n * d_c * d_h * y_w * dy_data_size;

        if(!dy_on_device){
                for(i = 0; i < num_pipeline; i++) cudaMalloc((void**)&d_dy[i], d_dy_size);
        }

        cudnnTensorDescriptor_t d_dyDesc;
        cudnnCreateTensorDescriptor(&d_dyDesc);


        // ConvolutionBackwardBias loop
        int i_n, i_c, i_h;

        int n_count, c_count, h_count;

        float one = 1.0f;  // ToDo : floatでいいのか
        float zero = 0.0f;

        void *d_beta;

        // initialize n_count
        n_count = 0;

        for(i_n = 0; i_n < n_step; i_n++){

                d_n = n / n_step;
                if(i_n < n % n_step) d_n++;

                // initialize c_count
                c_count = 0;

                for(i_c = 0; i_c < c_step; i_c++){

                        d_c = y_c / c_step;
                        if(i_c < y_c % c_step) d_c++;

                        // initialize h_count
                        h_count = 0;

                        for(i_h = 0; i_h < h_step; i_h++){

                                d_h = y_h / h_step;
                                if(i_h < y_h % h_step) d_h++;

                                // update d_dy
                                if(dy_on_device){
                                        cudnnSetTensor4dDescriptorEx(d_dyDesc, dy_dataType, d_n, d_c, d_h, y_w, nStride, y_cStride, y_hStride, y_wStride);
                                        d_dy[current] = (void *)dy + (n_count * nStride + c_count * y_cStride + h_count * y_hStride) * dy_data_size;
                                }
                                else{
                                        cudnnSetTensor4dDescriptor(d_dyDesc, dy_format, dy_dataType, d_n, d_c, d_h, y_w);
                                        ooc_cudnnGetSubTensor4D(handle, dyDesc, dy, d_dyDesc, d_dy[current], n_count, c_count, h_count, 0);
                                }



                                // update d_db
                                cudnnSetTensor4dDescriptor(d_dbDesc, CUDNN_TENSOR_NCHW, db_dataType, 1, d_c, 1, 1);
                                if(db_on_device){
                                        d_db[current] = db + (c_count * b_cStride) * db_data_size;
                                }
                                else{
                                        cudaStreamWaitEvent(streams[current], flags[(current + num_pipeline - 1) % num_pipeline], 0);
                                        ooc_cudnnGetSubTensor4D(handle, dbDesc, db, d_dbDesc, d_db[current], 0, c_count, 0, 0);
                                }


                                // convolution main
                                if((i_n == 0) && (i_h == 0)) d_beta = &zero;
                                else d_beta = &one;
                                
                                status = cudnnConvolutionBackwardBias(handle, alpha, d_dyDesc, d_dy[current], d_beta, d_dbDesc, d_db[current]);
                                

                                // update h_db
                                if(!db_on_device){
                                        ooc_cudnnSetSubTensor4D(handle, dbDesc, db, d_dbDesc, d_db[current], 0, c_count, 0, 0);
                                        cudaEventRecord(flags[current], streams[current]);
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

        for(i = 0; i < 3; i++) cudaEventDestroy(flags[i]);


        cudnnDestroyTensorDescriptor(d_dbDesc);
        cudnnDestroyTensorDescriptor(d_dyDesc);


        if(!dy_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_dy[i]);
        if(!db_on_device) for(i = 0; i < num_pipeline; i++) cudaFree(d_db[i]);

        return status;
}