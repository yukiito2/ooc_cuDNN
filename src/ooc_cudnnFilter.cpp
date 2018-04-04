
#include "ooc_cudnn.h"


cudnnStatus_t CUDNNWINAPI cudnnScaleFilter(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       wDesc,
                                void                               *w,
                                const void                         *alpha ){

        cudnnStatus_t status;

        // get parameter
        cudnnDataType_t w_dataType;
        cudnnTensorFormat_t w_format;
        int w_k, w_c, w_h, w_w;
        cudnnGetFilter4dDescriptor(wDesc, &w_dataType, &w_format, &w_k, &w_c, &w_h, &w_w);

        cudnnTensorDescriptor_t new_wDesc;
        cudnnCreateTensorDescriptor(&new_wDesc);

        cudnnSetTensor4dDescriptor(new_wDesc, w_format, w_dataType, w_k, w_c, w_h, w_w);

        status = cudnnScaleTensor(handle, new_wDesc, w, alpha);
        
        return status;
}



cudnnStatus_t CUDNNWINAPI ooc_cudnnPrintFilter4D(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       filterDesc,
                                const void                         *filter ){

        cudaStream_t stream;
        cudnnGetStream(handle, &stream);
        cudaStreamSynchronize(stream);

        // get parameter
        cudnnDataType_t filter_dataType;
        cudnnTensorFormat_t filter_format;
        int filter_k, filter_c, filter_h, filter_w;
        cudnnGetFilter4dDescriptor(filterDesc, &filter_dataType, &filter_format, &filter_k, &filter_c, &filter_h, &filter_w);

        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, filter);

        
        // print filter
        size_t data_size = cudnnSizeOf(filter_dataType);

        void *filter_ptr = (void *)filter;
        if(attr.memoryType == cudaMemoryTypeDevice){
                cudaMallocHost((void**)&filter_ptr, filter_k * filter_c * filter_h * filter_w * data_size);
                cudaMemcpy(filter_ptr, filter, filter_k * filter_c * filter_h * filter_w * data_size, cudaMemcpyDefault);
        }

        int k_count, c_count, h_count, w_count;

        for(k_count = 0; k_count < filter_k; k_count++){
                for(c_count = 0; c_count < filter_c; c_count++){
                        for(h_count = 0; h_count < filter_h; h_count++){
                                for(w_count = 0; w_count < filter_w; w_count++){
                                        if(filter_dataType == CUDNN_DATA_FLOAT) printf("%f ", *(float *)filter_ptr);
                                        else printf("%lf ", *(double *)filter);

                                        filter_ptr += data_size;
                                }
                                printf("\n");
                        }
                        printf("\n");
                }
                printf("\n");
        }

        if(attr.memoryType == cudaMemoryTypeDevice) cudaFree(filter_ptr);

        return CUDNN_STATUS_SUCCESS;
}


cudnnStatus_t CUDNNWINAPI ooc_cudnnGetSubFilter4D(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       filterDesc,
                                void                               *filter,
                                const cudnnFilterDescriptor_t       sub_filterDesc,
                                void                               *sub_filter,
                                const int                           k_offset,
                                const int                           c_offset ){


        cudaStream_t stream;
        cudnnGetStream(handle, &stream);

        // get parameter
        cudnnDataType_t filter_dataType;
        cudnnTensorFormat_t filter_format;
        int filter_k, filter_c, filter_h, filter_w;
        cudnnGetFilter4dDescriptor(filterDesc, &filter_dataType, &filter_format, &filter_k, &filter_c, &filter_h, &filter_w);

        cudnnDataType_t sub_filter_dataType;
        cudnnTensorFormat_t sub_filter_format;
        int sub_filter_k, sub_filter_c, sub_filter_h, sub_filter_w;
        cudnnGetFilter4dDescriptor(sub_filterDesc, &sub_filter_dataType, &sub_filter_format, &sub_filter_k, &sub_filter_c, &sub_filter_h, &sub_filter_w);

        // check parameter
        if(!(filter_dataType == sub_filter_dataType) || !(filter_format == sub_filter_format) || !(filter_h == sub_filter_h) || !(filter_w == sub_filter_w)){
                printf("cudnnGetSubFilter4D : CUDNN_STATUS_BAD_PARAM\n");
                return CUDNN_STATUS_BAD_PARAM;
        }

        // get sub filter
        size_t one_filter_size = filter_h * filter_w * cudnnSizeOf(filter_dataType);
        size_t buffer_size = sub_filter_c * one_filter_size;

        void *filter_ptr = filter + (k_offset * filter_c + c_offset) * one_filter_size;
        void *sub_filter_ptr = sub_filter;

        int k_count;

        for(k_count = 0; k_count < sub_filter_k; k_count++){

                cudaMemcpyAsync(sub_filter_ptr, filter_ptr, buffer_size, cudaMemcpyDefault, stream);

                filter_ptr += filter_c * one_filter_size;
                sub_filter_ptr += buffer_size;
        }

        return CUDNN_STATUS_SUCCESS;
}



// mode = 0 -> get, mode = 1 -> set
cudnnStatus_t CUDNNWINAPI ooc_cudnnCpySubFilter4D(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       filterDesc,
                                void                               *filter,
                                const cudnnFilterDescriptor_t       sub_filterDesc,
                                void                               *sub_filter,
                                const int                           k_offset,
                                const int                           c_offset,
                                const int                           h_offset,
                                const int                           w_offset,
                                const int                           mode ){

        
        cudaStream_t stream;
        cudnnGetStream(handle, &stream);


        // get parameter
        cudnnDataType_t filter_dataType;
        cudnnTensorFormat_t filter_format;
        int filter_k, filter_c, filter_h, filter_w;
        cudnnGetFilter4dDescriptor(filterDesc, &filter_dataType, &filter_format, &filter_k, &filter_c, &filter_h, &filter_w);

        cudnnDataType_t sub_filter_dataType;
        cudnnTensorFormat_t sub_filter_format;
        int sub_filter_k, sub_filter_c, sub_filter_h, sub_filter_w;
        cudnnGetFilter4dDescriptor(sub_filterDesc, &sub_filter_dataType, &sub_filter_format, &sub_filter_k, &sub_filter_c, &sub_filter_h, &sub_filter_w);


        // check parameter
        if(!(filter_dataType == sub_filter_dataType) || !(filter_format == sub_filter_format) ||
           (filter_k < sub_filter_k + k_offset) || (filter_c < sub_filter_c + c_offset) ||
           (filter_h < sub_filter_h + h_offset) || (filter_w < sub_filter_w + w_offset) ){
                printf("cudnnCpySubFilter4D : CUDNN_STATUS_BAD_PARAM\n");
                return CUDNN_STATUS_BAD_PARAM;
        }

        // get sub tensor
        bool w_bool = (filter_w == sub_filter_w);
        bool h_bool = (filter_h == sub_filter_h);
        bool c_bool = (filter_c == sub_filter_c);

        size_t data_size = cudnnSizeOf(filter_dataType);
        size_t filter_kStride = filter_c * filter_h * filter_w;
        size_t filter_cStride = filter_h * filter_w;
        size_t filter_hStride = filter_w;
        size_t filter_wStride = 1;

        size_t buffer_size;
        size_t filter_pitch;


        void *filter_ptr, *sub_filter_ptr;

        int k_count, c_count, h_count;

        if(w_bool){
                if(h_bool){
                        if(c_bool){
                                buffer_size = sub_filter_k * sub_filter_c * sub_filter_h * sub_filter_w * data_size;

                                filter_ptr = filter + (k_offset * filter_kStride) * data_size;
                                sub_filter_ptr = sub_filter;

                                if(mode == 0) cudaMemcpyAsync(sub_filter_ptr, filter_ptr, buffer_size, cudaMemcpyDefault, stream);
                                else cudaMemcpyAsync(filter_ptr, sub_filter_ptr, buffer_size, cudaMemcpyDefault, stream);
                        }
                        else{
                                buffer_size = sub_filter_c * sub_filter_h * sub_filter_w * data_size;
                                filter_pitch = filter_kStride * data_size;

                                filter_ptr = filter + (k_offset * filter_kStride + c_offset * filter_cStride) * data_size;
                                sub_filter_ptr = sub_filter;

                                if(mode == 0) cudaMemcpy2DAsync(sub_filter_ptr, buffer_size, filter_ptr, filter_pitch, buffer_size, sub_filter_k, cudaMemcpyDefault, stream);
                                else cudaMemcpy2DAsync(filter_ptr, filter_pitch, sub_filter_ptr, buffer_size, buffer_size, sub_filter_k, cudaMemcpyDefault, stream);
                        }
                }
                else{   
                        buffer_size = sub_filter_h * sub_filter_w * data_size;
                        filter_pitch = filter_cStride * data_size;

                        for(k_count = 0; k_count < sub_filter_k; k_count++){
                                filter_ptr = filter + ((k_count + k_offset) * filter_kStride + c_offset * filter_cStride + h_offset * filter_hStride) * data_size;
                                sub_filter_ptr = sub_filter + k_count * sub_filter_c * buffer_size;

                                if(mode == 0) cudaMemcpy2DAsync(sub_filter_ptr, buffer_size, filter_ptr, filter_pitch, buffer_size, sub_filter_c, cudaMemcpyDefault, stream);
                                else cudaMemcpy2DAsync(filter_ptr, filter_pitch, sub_filter_ptr, buffer_size, buffer_size, sub_filter_c, cudaMemcpyDefault, stream);
                        }
                }
        }
        else{        
                buffer_size = sub_filter_w * data_size;
                filter_pitch = filter_hStride * data_size;

                for(k_count = 0; k_count < sub_filter_k; k_count++){
                        for(c_count = 0; c_count < sub_filter_c; c_count++){
                                filter_ptr = filter + ((k_count + k_offset) * filter_kStride + (c_count + c_offset) * filter_cStride + h_offset * filter_hStride + w_offset * filter_wStride) * data_size;
                                sub_filter_ptr = sub_filter + (k_count * sub_filter_c * sub_filter_h + c_count * sub_filter_h) * buffer_size;

                                if(mode == 0) cudaMemcpy2DAsync(sub_filter_ptr, buffer_size, filter_ptr, filter_pitch, buffer_size, sub_filter_h, cudaMemcpyDefault, stream);
                                else cudaMemcpy2DAsync(filter_ptr, filter_pitch, sub_filter_ptr, buffer_size, buffer_size, sub_filter_h, cudaMemcpyDefault, stream);
                        }
                }
        }

        return CUDNN_STATUS_SUCCESS;
}


cudnnStatus_t CUDNNWINAPI ooc_cudnnGetSubFilter4D(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       filterDesc,
                                const void                         *filter,
                                const cudnnFilterDescriptor_t       sub_filterDesc,
                                void                               *sub_filter,
                                const int                           k_offset,
                                const int                           c_offset,
                                const int                           h_offset,
                                const int                           w_offset ){

        cudnnStatus_t status;

        status = ooc_cudnnCpySubFilter4D(handle, filterDesc, (void *)filter, sub_filterDesc, sub_filter, k_offset, c_offset, h_offset, w_offset, 0);

        return status;
}


cudnnStatus_t CUDNNWINAPI ooc_cudnnSetSubFilter4D(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       filterDesc,
                                void                               *filter,
                                const cudnnFilterDescriptor_t       sub_filterDesc,
                                const void                         *sub_filter,
                                const int                           k_offset,
                                const int                           c_offset,
                                const int                           h_offset,
                                const int                           w_offset ){

        cudnnStatus_t status;

        status = ooc_cudnnCpySubFilter4D(handle, filterDesc, filter, sub_filterDesc, (void *)sub_filter, k_offset, c_offset, h_offset, w_offset, 1);

        return status;
}