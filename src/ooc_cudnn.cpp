
#include "ooc_cudnn.h"
#include "ooc_cuda.h"
#include "ooc_cublas.h"

bool ooc_cudnnInit_flag = false;

void *ooc_cudnnWorkspace;
size_t ooc_cudnnWorkspace_size = 0;

size_t CUDNNWINAPI cudnnSizeOf(cudnnDataType_t dataType){
        size_t size = 0;

        if(dataType == CUDNN_DATA_FLOAT) size = sizeof(float);
        else if(dataType == CUDNN_DATA_DOUBLE) size = sizeof(double);
        else if(dataType == CUDNN_DATA_HALF) size = sizeof(float) / 2;

        return size;
}


cudnnStatus_t CUDNNWINAPI ooc_cudnnInit(){

		ooc_cudnnAddTensorProfile();
		ooc_cudnnConvolutionForwardProfile();
		ooc_cudnnConvolutionBackwardDataProfile();
		ooc_cudnnConvolutionBackwardFilterProfile();
		ooc_cudnnConvolutionBackwardBiasProfile();
		ooc_cudnnActivationForwardProfile();
		ooc_cudnnActivationBackwardProfile();
		ooc_cudnnPoolingForwardProfile();
		ooc_cudnnPoolingBackwardProfile();
		ooc_cublasSgemmProfile();
		ooc_cudnnSoftmaxForwardProfile();
		ooc_cudnnSoftmaxBackwardProfile();
		ooc_cudaMemcpyProfile();

		ooc_cudnnInit_flag = true;

		return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t CUDNNWINAPI ooc_cudnnCreate(cudnnHandle_t *handle){
		if(ooc_cudnnInit_flag == false) ooc_cudnnInit();
		return cudnnCreate(handle);
}