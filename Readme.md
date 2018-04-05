ooc_cuDNN: out-of-core cuDNN
====

The objective of out-of-core cuDNN (ooc_cuDNN) library is to compute large neural networks exceeding GPU memory capacity. ooc_cuDNN is an extention of [cuDNN](https://arxiv.org/abs/1410.0759) library and compatible with cuDNN. For memory management, ooc_cuDNN also includes extensions of some CUDA functions.

The basic design policies of ooc_cuDNN are as follows:
+ Data that exceeds GPU memory capacity is placed in CPU memory.
+ CNN computations are performed with GPU using the original cuDNN functions
+ Layers and filters are divided for each dimension and computed.
+ When data in the CPU memory is required for computation, communication between CPU-GPU is performed automatically.
+ CPU-GPU communication is overlapped with computation.

For performance improvement, division sizes of each data are optimized based on performance model.  
In addition, to reduce extra communication, ooc_cuDNN provides fused functions that perform multiple CNN computations at a time. (These fused functions is not provided in original cuDNN.)

## Requirement
+ [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit)
+ [cuDNN v7](https://developer.nvidia.com/cudnn)

## Usage
+ `build.sh` makes `include/{ooc_cudnn.h, ooc_cuda.h, ooc_cublas.h}` and `lib64/{libooc_cudnn.so, libooc_cuda.so, libooc_cublas.so}`.
+ `clean.sh` cleans those.
<br />

+ Edit your application code by adding `#include <ooc_cudnn.h>`, `#include <ooc_cuda.h>` and `#include <ooc_cublas.h>`. 
+ And, replace cudnn-APIs with ooc_cuDNN-APIs. (Only add `ooc_` to head of each cuDNN functions.)
+ Do the same for CUDA runtime-APIs and cuBLAS-APIs.
+ Note: current ooc_cuDNN support only some cuDNN, CUDA and cuBLAS functions. So, check which functions is supported by looking at `ooc_cudnn.h` etc.

example: use original cuDNN
```c
cudaMalloc(...);  
cudaMemcpy(...);  
cudnnConvolutionForward(...);  
cudnnAddBias(...);  
cudnnActivationForward(...);  
cublasSgemm(...);  
```
=> example: use ooc_cuDNN
```c
ooc_cudaMalloc(...);  
ooc_cudaMemcpy(...);  
ooc_cudnnConvolutionForward(...);  
ooc_cudnnAddBias(...);  
ooc_cudnnActivationForward(...);  
ooc_cublasSgemm(...); 
```

+ Additional:
	+ To use ooc_cuDNN fused functions, you need to modify code as following. 
	+ Note: Check the arguments of fused functions by looking at `ooc_cudnn.h`.

example: not use fused functions
```c
ooc_cudnnConvolutionForward(...);  
ooc_cudnnAddBias(...);  
ooc_cudnnActivationForward(...); 
```
=> example: use fused functions
```c
ooc_cudnnConvolutionBiasActivationForward(...);  
```

## Current limitations
+ Current ooc_cuDNN is an extention of cuDNNv7. So, code modification may increase in case that your original code uses other versions.
+ Current ooc_cuDNN only supports "float" data-type and "NCHW" data-format. 
+ Using CUDA kernels other than ooc_cuDNN may cause errors, since ooc_cudaMalloc() allocate some data to host-memory instead of device-memory.

## References

+ Yuki Ito, Ryo Matsumiya, and Toshio Endo. ooc_cuDNN: Accommodating Convolutional Neural Networks over GPU Memory Capacity. In Proceedings of 2017 IEEE International Conference on Big Data (IEEE BigData 2017), Boston, December 2017.

## Acknowledgements

This work is supported by JST-CREST, “Software Technology that Deals with Deeper Memory Hierarchy in Postpetascale Era”.

## Author

[yukiito](https://github.com/yukiito2)

Copyright (C) 2017, Yuki Ito. All Rights Reserved.
