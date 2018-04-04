
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#include <cuda.h>
#include <cudnn.h>


// debug
#define CUDA_SAFE_CALL(func) \
do  { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
         exit(err); \
    } \
} while(0)



size_t CUDNNWINAPI cudnnSizeOf(cudnnDataType_t dataType);


cudnnStatus_t CUDNNWINAPI ooc_cudnnCreate(cudnnHandle_t *handle);


cudnnStatus_t CUDNNWINAPI ooc_cudnnInit();


cudnnStatus_t CUDNNWINAPI ooc_cudnnPrintTensor4D(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       tensorDesc,
                                const void                         *tensor );


cudnnStatus_t CUDNNWINAPI ooc_cudnnGetSubTensor4D(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       tensorDesc,
                                const void                         *tensor,
                                const cudnnTensorDescriptor_t       sub_tensorDesc,
                                void                               *sub_tensor,
                                const size_t                        n_offset,
                                const size_t                        c_offset,
                                const size_t                        h_offset,
                                const size_t                        w_offset );

cudnnStatus_t CUDNNWINAPI ooc_cudnnSetSubTensor4D(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       tensorDesc,
                                void                               *tensor,
                                const cudnnTensorDescriptor_t       sub_tensorDesc,
                                const void                         *sub_tensor,
                                const size_t                        n_offset,
                                const size_t                        c_offset,
                                const size_t                        h_offset,
                                const size_t                        w_offset );


cudnnStatus_t CUDNNWINAPI cudnnScaleFilter(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       wDesc,
                                void                               *w,
                                const void                         *alpha );


cudnnStatus_t CUDNNWINAPI ooc_cudnnPrintFilter4D(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       filterDesc,
                                const void                         *filter );



cudnnStatus_t CUDNNWINAPI ooc_cudnnGetSubFilter4D(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       filterDesc,
                                const void                         *filter,
                                const cudnnFilterDescriptor_t       sub_filterDesc,
                                void                               *sub_filter,
                                const int                           k_offset,
                                const int                           c_offset,
                                const int                           h_offset,
                                const int                           w_offset );



cudnnStatus_t CUDNNWINAPI ooc_cudnnSetSubFilter4D(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       filterDesc,
                                void                               *filter,
                                const cudnnFilterDescriptor_t       sub_filterDesc,
                                const void                         *sub_filter,
                                const int                           k_offset,
                                const int                           c_offset,
                                const int                           h_offset,
                                const int                           w_offset );


/* Function to perform the forward pass for batch add bias */
cudnnStatus_t CUDNNWINAPI ooc_cudnnAddTensorProfile();


cudnnStatus_t CUDNNWINAPI ooc_cudnnAddTensor(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       aDesc,
                                const void                         *A,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       cDesc,
                                void                               *C );


/* Function to perform the forward pass for batch convolution */
cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionForwardProfile();


cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionForward(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionFwdAlgo_t           algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );


/* Function to perform the forward pass for activation */
cudnnStatus_t CUDNNWINAPI ooc_cudnnActivationForwardProfile();


cudnnStatus_t CUDNNWINAPI ooc_cudnnActivationForward(
                                cudnnHandle_t                       handle,
                                cudnnActivationDescriptor_t         activationDesc,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );


/* Function to perform the backward pass for activation */
cudnnStatus_t CUDNNWINAPI ooc_cudnnActivationBackwardProfile();


cudnnStatus_t CUDNNWINAPI ooc_cudnnActivationBackward(
                                cudnnHandle_t                       handle,
                                cudnnActivationDescriptor_t         activationDesc,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx );


// convolution + add bias + activation
cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBiasActivationForward(
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
                                void                               *y );


/* Function to perform the backward pass for batch convolution */
cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBackwardDataProfile();


cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBackwardData(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionBwdDataAlgo_t       algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx );


cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBackwardBiasProfile();


cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBackwardBias(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dbDesc,
                                void                               *db );


cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBackwardFilterProfile();


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
                                void                               *dw );


cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBackwardDataFilterBias(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
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
                                void                               *db );


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
                                void                               *db );


/* Function to perform the Forward pass for batch deconvolution */
cudnnStatus_t CUDNNWINAPI ooc_cudnnDeconvolutionBiasActivationForward(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionBwdDataAlgo_t       algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const cudnnTensorDescriptor_t       biasDesc,
                                const void                         *bias,
                                cudnnActivationDescriptor_t         activationDesc,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );


/* Function to perform the Forward pass for pooling */
cudnnStatus_t CUDNNWINAPI ooc_cudnnPoolingForwardProfile();


cudnnStatus_t CUDNNWINAPI ooc_cudnnPoolingForward(
                                cudnnHandle_t                       handle,
                                cudnnPoolingDescriptor_t            poolingDesc,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );


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
                                void                               *z );


cudnnStatus_t CUDNNWINAPI ooc_cudnnPoolingActivationConvolutionBackwardDataFilterBias(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnTensorDescriptor_t       zDesc,
                                const void                         *z,
                                const cudnnTensorDescriptor_t       dzDesc,
                                const void                         *dz,
                                const cudnnPoolingDescriptor_t      poolingDesc,
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
                                void                               *db );


/* Function to perform the Backward pass for pooling */
cudnnStatus_t CUDNNWINAPI ooc_cudnnPoolingBackwardProfile();


cudnnStatus_t CUDNNWINAPI ooc_cudnnPoolingBackward(
                                cudnnHandle_t                       handle,
                                const cudnnPoolingDescriptor_t      poolingDesc,
                                const void                          *alpha,
                                const cudnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx );


/* Function to perform the Forward pass for softmax */
cudnnStatus_t CUDNNWINAPI ooc_cudnnSoftmaxForwardProfile();


cudnnStatus_t CUDNNWINAPI ooc_cudnnSoftmaxForward(
                                cudnnHandle_t                       handle,
                                cudnnSoftmaxAlgorithm_t             algo,
                                cudnnSoftmaxMode_t                  mode,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );


/* Function to perform the Backward pass for softmax */
cudnnStatus_t CUDNNWINAPI ooc_cudnnSoftmaxBackwardProfile();


cudnnStatus_t CUDNNWINAPI ooc_cudnnSoftmaxBackward(
                                cudnnHandle_t                       handle,
                                cudnnSoftmaxAlgorithm_t             algo,
                                cudnnSoftmaxMode_t                  mode,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx );


