
#include "ooc_cudnn.h"



cudnnStatus_t CUDNNWINAPI ooc_cudnnPrintTensor4D(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       tensorDesc,
                                const void                         *tensor ){

        cudaStream_t stream;
        cudnnGetStream(handle, &stream);
        cudaStreamSynchronize(stream);

        // get parameter
        cudnnDataType_t tensor_dataType;
        cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
        int tensor_n, tensor_c, tensor_h, tensor_w, tensor_nStride_tmp, tensor_cStride_tmp, tensor_hStride_tmp, tensor_wStride_tmp;
        cudnnGetTensor4dDescriptor(tensorDesc, &tensor_dataType, &tensor_n, &tensor_c, &tensor_h, &tensor_w, &tensor_nStride_tmp, &tensor_cStride_tmp, &tensor_hStride_tmp, &tensor_wStride_tmp);
        size_t tensor_nStride = (size_t)tensor_c * (size_t)tensor_h * (size_t)tensor_w, tensor_cStride = (size_t)tensor_h * (size_t)tensor_w, tensor_hStride = (size_t)tensor_w, tensor_wStride = 1;
        

        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, tensor);

        // print tensor
        size_t data_size = cudnnSizeOf(tensor_dataType);
        
        void *tensor_ptr = (void *)tensor;
        if(attr.memoryType == cudaMemoryTypeDevice){
                cudaMallocHost((void**)&tensor_ptr, tensor_n * tensor_nStride * data_size);
                cudaMemcpy(tensor_ptr, tensor, tensor_n * tensor_nStride * data_size, cudaMemcpyDefault);
        }


        int n_count, c_count, h_count, w_count;
        size_t offset;
    
        printf("ooc_cudnnPrintTensor4D : n -> %d, c -> %d, h -> %d, w -> %d\n", tensor_n ,tensor_c ,tensor_h ,tensor_w);
        printf("ooc_cudnnPrintTensor4D : nStride -> %d, cStride -> %d, hStride -> %d, wStride -> %d\n", tensor_nStride ,tensor_cStride ,tensor_hStride ,tensor_wStride);
        
        for(n_count = 0; n_count < tensor_n; n_count++){
                for(c_count = 0; c_count < tensor_c; c_count++){
                        for(h_count = 0; h_count < tensor_h; h_count++){
                                for(w_count = 0; w_count < tensor_w; w_count++){
                                        offset = n_count * tensor_nStride + c_count * tensor_cStride + h_count * tensor_hStride + w_count * tensor_wStride;

                                        if(tensor_dataType == CUDNN_DATA_FLOAT) printf("%f ", ((float *)tensor_ptr)[offset]);
                                        else printf("%lf \n", ((double *)tensor_ptr)[offset]);
                                }
                                printf("\n");
                        }
                        printf("\n");
                }
                printf("\n");
        }

        if(attr.memoryType == cudaMemoryTypeDevice) cudaFree(tensor_ptr);

        return CUDNN_STATUS_SUCCESS;
}


// mode = 0 -> get, mode = 1 -> set
cudnnStatus_t CUDNNWINAPI ooc_cudnnCpySubTensor4D(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       tensorDesc,
                                void                               *tensor,
                                const cudnnTensorDescriptor_t       sub_tensorDesc,
                                void                               *sub_tensor,
                                const size_t                        n_offset,
                                const size_t                        c_offset,
                                const size_t                        h_offset,
                                const size_t                        w_offset,
                                const size_t                        mode ){

        
        cudaStream_t stream;
        cudnnGetStream(handle, &stream);

        
        // get parameter
        cudnnDataType_t tensor_dataType;
        cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
        int tensor_n, tensor_c, tensor_h, tensor_w, tensor_nStride_tmp, tensor_cStride_tmp, tensor_hStride_tmp, tensor_wStride_tmp;
        cudnnGetTensor4dDescriptor(tensorDesc, &tensor_dataType, &tensor_n, &tensor_c, &tensor_h, &tensor_w, &tensor_nStride_tmp, &tensor_cStride_tmp, &tensor_hStride_tmp, &tensor_wStride_tmp);
        size_t tensor_nStride = (size_t)tensor_c * (size_t)tensor_h * (size_t)tensor_w, tensor_cStride = (size_t)tensor_h * (size_t)tensor_w, tensor_hStride = (size_t)tensor_w, tensor_wStride = 1;

        cudnnDataType_t sub_tensor_dataType;
        cudnnTensorFormat_t sub_tensor_format = CUDNN_TENSOR_NCHW;
        int sub_tensor_n, sub_tensor_c, sub_tensor_h, sub_tensor_w, sub_tensor_nStride_tmp, sub_tensor_cStride_tmp, sub_tensor_hStride_tmp, sub_tensor_wStride_tmp;
        cudnnGetTensor4dDescriptor(sub_tensorDesc, &sub_tensor_dataType, &sub_tensor_n, &sub_tensor_c, &sub_tensor_h, &sub_tensor_w, &sub_tensor_nStride_tmp, &sub_tensor_cStride_tmp, &sub_tensor_hStride_tmp, &sub_tensor_wStride_tmp);
        size_t sub_tensor_nStride = (size_t)sub_tensor_c * (size_t)sub_tensor_h * (size_t)sub_tensor_w, sub_tensor_cStride = (size_t)sub_tensor_h * (size_t)sub_tensor_w, sub_tensor_hStride = (size_t)sub_tensor_w, sub_tensor_wStride = 1;


        // check parameter
        if(!(tensor_dataType == sub_tensor_dataType) || !(tensor_format == sub_tensor_format)){
                printf("cudnnCpySubTensor4D : CUDNN_STATUS_BAD_PARAM\n");
                return CUDNN_STATUS_BAD_PARAM;
        }

        // compute copy_range
        size_t tensor_n_range[2], sub_tensor_n_range[2];
        tensor_n_range[0] = n_offset;
        tensor_n_range[1] = tensor_n_range[0] + sub_tensor_n - 1;

        if(tensor_n_range[0] < 0) tensor_n_range[0] = 0;
        else if(tensor_n_range[0] > tensor_n - 1) tensor_n_range[0] = tensor_n;
        if(tensor_n_range[1] < 0) tensor_n_range[1] = -1;
        else if(tensor_n_range[1] > tensor_n - 1) tensor_n_range[1] = tensor_n - 1;

        sub_tensor_n_range[0] = tensor_n_range[0] - n_offset;
        sub_tensor_n_range[1] = sub_tensor_n_range[0] + tensor_n_range[1] - tensor_n_range[0];


        size_t tensor_c_range[2], sub_tensor_c_range[2];
        tensor_c_range[0] = c_offset;
        tensor_c_range[1] = tensor_c_range[0] + sub_tensor_c - 1;

        if(tensor_c_range[0] < 0) tensor_c_range[0] = 0;
        else if(tensor_c_range[0] > tensor_c - 1) tensor_c_range[0] = tensor_c;
        if(tensor_c_range[1] < 0) tensor_c_range[1] = -1;
        else if(tensor_c_range[1] > tensor_c - 1) tensor_c_range[1] = tensor_c - 1;

        sub_tensor_c_range[0] = tensor_c_range[0] - c_offset;
        sub_tensor_c_range[1] = sub_tensor_c_range[0] + tensor_c_range[1] - tensor_c_range[0];


        size_t tensor_h_range[2], sub_tensor_h_range[2];
        tensor_h_range[0] = h_offset;
        tensor_h_range[1] = tensor_h_range[0] + sub_tensor_h - 1;

        if(tensor_h_range[0] < 0) tensor_h_range[0] = 0;
        else if(tensor_h_range[0] > tensor_h - 1) tensor_h_range[0] = tensor_h;
        if(tensor_h_range[1] < 0) tensor_h_range[1] = -1;
        else if(tensor_h_range[1] > tensor_h - 1) tensor_h_range[1] = tensor_h - 1;
        
        sub_tensor_h_range[0] = tensor_h_range[0] - h_offset;
        sub_tensor_h_range[1] = sub_tensor_h_range[0] + tensor_h_range[1] - tensor_h_range[0];

        
        size_t tensor_w_range[2], sub_tensor_w_range[2];
        tensor_w_range[0] = w_offset;
        tensor_w_range[1] = tensor_w_range[0] + sub_tensor_w - 1;

        if(tensor_w_range[0] < 0) tensor_w_range[0] = 0;
        else if(tensor_w_range[0] > tensor_w - 1) tensor_w_range[0] = tensor_w;
        if(tensor_w_range[1] < 0) tensor_w_range[1] = -1;
        else if(tensor_w_range[1] > tensor_w - 1) tensor_w_range[1] = tensor_w - 1;
        
        sub_tensor_w_range[0] = tensor_w_range[0] - w_offset;
        sub_tensor_w_range[1] = sub_tensor_w_range[0] + tensor_w_range[1] - tensor_w_range[0];


        // copy tensor
        size_t n_length = tensor_n_range[1] - tensor_n_range[0] + 1;
        size_t c_length = tensor_c_range[1] - tensor_c_range[0] + 1;
        size_t h_length = tensor_h_range[1] - tensor_h_range[0] + 1;
        size_t w_length = tensor_w_range[1] - tensor_w_range[0] + 1;

        bool w_bool = (tensor_w_range[0] == 0) && (tensor_w_range[1] == tensor_w - 1) && (sub_tensor_w_range[0] == 0) && (sub_tensor_w_range[1] == sub_tensor_w - 1);
        bool h_bool = (tensor_h_range[0] == 0) && (tensor_h_range[1] == tensor_h - 1) && (sub_tensor_h_range[0] == 0) && (sub_tensor_h_range[1] == sub_tensor_h - 1);
        bool c_bool = (tensor_c_range[0] == 0) && (tensor_c_range[1] == tensor_c - 1) && (sub_tensor_c_range[0] == 0) && (sub_tensor_c_range[1] == sub_tensor_c - 1);
        bool n_bool = (tensor_n_range[0] == 0) && (tensor_n_range[1] == tensor_n - 1) && (sub_tensor_n_range[0] == 0) && (sub_tensor_n_range[1] == sub_tensor_n - 1);

        size_t data_size = cudnnSizeOf(tensor_dataType);
        size_t buffer_size;
        size_t tensor_pitch, sub_tensor_pitch;


        void *tensor_ptr, *sub_tensor_ptr;

        size_t n_count, c_count, h_count;

        if(w_bool){
                if(h_bool){
                        if(c_bool){
                                buffer_size = n_length * c_length * h_length * w_length * data_size;

                                tensor_ptr = tensor + (tensor_n_range[0] * tensor_nStride) * data_size;
                                sub_tensor_ptr = sub_tensor + (sub_tensor_n_range[0] * sub_tensor_nStride) * data_size;

                                if(mode == 0) cudaMemcpyAsync(sub_tensor_ptr, tensor_ptr, buffer_size, cudaMemcpyDefault, stream);
                                else cudaMemcpyAsync(tensor_ptr, sub_tensor_ptr, buffer_size, cudaMemcpyDefault, stream);
                        }
                        else{
                                buffer_size = c_length * h_length * w_length * data_size;
                                tensor_pitch = tensor_nStride * data_size;
                                sub_tensor_pitch = sub_tensor_nStride * data_size;

                                tensor_ptr = tensor + (tensor_n_range[0] * tensor_nStride + tensor_c_range[0] * tensor_cStride) * data_size;
                                sub_tensor_ptr = sub_tensor + (sub_tensor_n_range[0] * sub_tensor_nStride + sub_tensor_c_range[0] * sub_tensor_cStride) * data_size;

                                if(mode == 0) cudaMemcpy2DAsync(sub_tensor_ptr, sub_tensor_pitch, tensor_ptr, tensor_pitch, buffer_size, n_length, cudaMemcpyDefault, stream);
                                else cudaMemcpy2DAsync(tensor_ptr, tensor_pitch, sub_tensor_ptr, sub_tensor_pitch, buffer_size, n_length, cudaMemcpyDefault, stream);
                        }
                }
                else{   
                        buffer_size = h_length * w_length * data_size;
                        tensor_pitch = tensor_cStride * data_size;
                        sub_tensor_pitch = sub_tensor_cStride * data_size;

                        if(c_bool){
                                tensor_ptr = tensor + (tensor_n_range[0] * tensor_nStride + tensor_h_range[0] * tensor_hStride) * data_size;
                                sub_tensor_ptr = sub_tensor + (sub_tensor_n_range[0] * sub_tensor_nStride + sub_tensor_h_range[0] * sub_tensor_hStride) * data_size;

                                if(mode == 0) cudaMemcpy2DAsync(sub_tensor_ptr, sub_tensor_pitch, tensor_ptr, tensor_pitch, buffer_size, n_length * c_length, cudaMemcpyDefault, stream);
                                else cudaMemcpy2DAsync(tensor_ptr, tensor_pitch, sub_tensor_ptr, sub_tensor_pitch, buffer_size, n_length * c_length, cudaMemcpyDefault, stream);
                        }
                        else{
                                for(n_count = 0; n_count < n_length; n_count++){
                                        tensor_ptr = tensor + ((n_count + tensor_n_range[0]) * tensor_nStride + tensor_c_range[0] * tensor_cStride + tensor_h_range[0] * tensor_hStride) * data_size;
                                        sub_tensor_ptr = sub_tensor + ((n_count + sub_tensor_n_range[0]) * sub_tensor_nStride + sub_tensor_c_range[0] * sub_tensor_cStride + sub_tensor_h_range[0] * sub_tensor_hStride) * data_size;

                                        if(mode == 0) cudaMemcpy2DAsync(sub_tensor_ptr, sub_tensor_pitch, tensor_ptr, tensor_pitch, buffer_size, c_length, cudaMemcpyDefault, stream);
                                        else cudaMemcpy2DAsync(tensor_ptr, tensor_pitch, sub_tensor_ptr, sub_tensor_pitch, buffer_size, c_length, cudaMemcpyDefault, stream);
                                }
                        }
                }
        }
        else{        
                buffer_size = w_length * data_size;
                tensor_pitch = tensor_hStride * data_size;
                sub_tensor_pitch = sub_tensor_hStride * data_size;

                for(n_count = 0; n_count < n_length; n_count++){
                        for(c_count = 0; c_count < c_length; c_count++){
                                tensor_ptr = tensor + ((n_count + tensor_n_range[0]) * tensor_nStride + (c_count + tensor_c_range[0]) * tensor_cStride + tensor_h_range[0] * tensor_hStride + tensor_w_range[0] * tensor_wStride) * data_size;
                                sub_tensor_ptr = sub_tensor + ((n_count + sub_tensor_n_range[0]) * sub_tensor_nStride + (c_count + sub_tensor_c_range[0]) * sub_tensor_cStride + sub_tensor_h_range[0] * sub_tensor_hStride + sub_tensor_w_range[0] * sub_tensor_wStride) * data_size;

                                if(mode == 0) cudaMemcpy2DAsync(sub_tensor_ptr, sub_tensor_pitch, tensor_ptr, tensor_pitch, buffer_size, h_length, cudaMemcpyDefault, stream);
                                else cudaMemcpy2DAsync(tensor_ptr, tensor_pitch, sub_tensor_ptr, sub_tensor_pitch, buffer_size, h_length, cudaMemcpyDefault, stream);
                        }
                }
        }


        // When mode is "Get", sub_tensor is filled by 0.
        if(mode == 0){
                // n 
                if(!n_bool){
                        buffer_size = sub_tensor_n_range[0] * sub_tensor_nStride * data_size;
                        sub_tensor_ptr = sub_tensor;
                        cudaMemsetAsync(sub_tensor_ptr, 0, buffer_size, stream);

                        buffer_size = (sub_tensor_n - sub_tensor_n_range[1] - 1) * sub_tensor_nStride * data_size;
                        sub_tensor_ptr = sub_tensor + (sub_tensor_n_range[1] + 1) * sub_tensor_nStride * data_size;
                        cudaMemsetAsync(sub_tensor_ptr, 0, buffer_size, stream);
                }

                // c
                if(!c_bool){
                        sub_tensor_pitch = sub_tensor_nStride * data_size;

                        buffer_size = sub_tensor_c_range[0] * sub_tensor_cStride * data_size;
                        sub_tensor_ptr = sub_tensor + sub_tensor_n_range[0] * sub_tensor_nStride * data_size;
                        cudaMemset2DAsync(sub_tensor_ptr, sub_tensor_pitch, 0, buffer_size, n_length, stream);

                        buffer_size = (sub_tensor_c - sub_tensor_c_range[1] - 1) * sub_tensor_cStride * data_size;
                        sub_tensor_ptr = sub_tensor + (sub_tensor_n_range[0] * sub_tensor_nStride + (sub_tensor_c_range[1] + 1) * sub_tensor_cStride) * data_size;
                        cudaMemset2DAsync(sub_tensor_ptr, sub_tensor_pitch, 0, buffer_size, n_length, stream);
                }

                // h
                if(!h_bool){
                        sub_tensor_pitch = sub_tensor_cStride * data_size;
                        
                        buffer_size = sub_tensor_h_range[0] * sub_tensor_hStride * data_size;
                        for(n_count = 0; n_count < n_length; n_count++){
                                sub_tensor_ptr = sub_tensor + ((n_count + sub_tensor_n_range[0]) * sub_tensor_nStride + sub_tensor_c_range[0] * sub_tensor_cStride) * data_size;
                                cudaMemset2DAsync(sub_tensor_ptr, sub_tensor_pitch, 0, buffer_size, c_length, stream);
                        }

                        buffer_size = (sub_tensor_h - sub_tensor_h_range[1] - 1) * sub_tensor_hStride * data_size;
                        for(n_count = 0; n_count < n_length; n_count++){
                                sub_tensor_ptr = sub_tensor + ((n_count + sub_tensor_n_range[0]) * sub_tensor_nStride + sub_tensor_c_range[0] * sub_tensor_cStride + (sub_tensor_h_range[1] + 1) * sub_tensor_hStride) * data_size;
                                cudaMemset2DAsync(sub_tensor_ptr, sub_tensor_pitch, 0, buffer_size, c_length, stream);
                        }
                }

                // w
                if(!w_bool){
                        sub_tensor_pitch = sub_tensor_hStride * data_size;

                        buffer_size = sub_tensor_w_range[0] * sub_tensor_wStride * data_size;
                        for(n_count = 0; n_count < n_length; n_count++){
                                for(c_count = 0; c_count < c_length; c_count++){
                                        sub_tensor_ptr = sub_tensor + ((n_count + sub_tensor_n_range[0]) * sub_tensor_nStride + (c_count + sub_tensor_c_range[0]) * sub_tensor_cStride + sub_tensor_h_range[0] * sub_tensor_hStride) * data_size;
                                        cudaMemset2DAsync(sub_tensor_ptr, sub_tensor_pitch, 0, buffer_size, h_length, stream);
                                }
                        }

                        buffer_size = (sub_tensor_w - sub_tensor_w_range[1] - 1) * sub_tensor_wStride * data_size;
                        for(n_count = 0; n_count < n_length; n_count++){
                                for(c_count = 0; c_count < c_length; c_count++){
                                        sub_tensor_ptr = sub_tensor + ((n_count + sub_tensor_n_range[0]) * sub_tensor_nStride + (c_count + sub_tensor_c_range[0]) * sub_tensor_cStride + sub_tensor_h_range[0] * sub_tensor_hStride + (sub_tensor_w_range[1] + 1) * sub_tensor_wStride) * data_size;
                                        cudaMemset2DAsync(sub_tensor_ptr, sub_tensor_pitch, 0, buffer_size, h_length, stream);
                                }
                        }
                }
        }


        return CUDNN_STATUS_SUCCESS;
}


cudnnStatus_t CUDNNWINAPI ooc_cudnnGetSubTensor4D(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       tensorDesc,
                                const void                         *tensor,
                                const cudnnTensorDescriptor_t       sub_tensorDesc,
                                void                               *sub_tensor,
                                const size_t                        n_offset,
                                const size_t                        c_offset,
                                const size_t                        h_offset,
                                const size_t                        w_offset ){

        cudnnStatus_t status;

        status = ooc_cudnnCpySubTensor4D(handle, tensorDesc, (void *)tensor, sub_tensorDesc, sub_tensor, n_offset, c_offset, h_offset, w_offset, 0);

        return status;
}


cudnnStatus_t CUDNNWINAPI ooc_cudnnSetSubTensor4D(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       tensorDesc,
                                void                               *tensor,
                                const cudnnTensorDescriptor_t       sub_tensorDesc,
                                const void                         *sub_tensor,
                                const size_t                        n_offset,
                                const size_t                        c_offset,
                                const size_t                        h_offset,
                                const size_t                        w_offset ){

        cudnnStatus_t status;

        status = ooc_cudnnCpySubTensor4D(handle, tensorDesc, tensor, sub_tensorDesc, (void *)sub_tensor, n_offset, c_offset, h_offset, w_offset, 1);

        return status;
}
