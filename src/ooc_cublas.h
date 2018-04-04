
#include <stdio.h>
#include <time.h>
#include <sys/time.h>


#include <cublas_v2.h>
#include <cublasXt.h>
#include <cublas_api.h>




// debug
#define CUDA_SAFE_CALL(func) \
do  { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
         exit(err); \
    } \
} while(0)


/* Function to perform sgemm */
cublasStatus_t CUBLASWINAPI ooc_cublasSgemmProfile();


cublasStatus_t CUBLASWINAPI ooc_cublasSgemm(
                                cublasHandle_t                      handle, 
                                cublasOperation_t                   transa, 
                                cublasOperation_t                   transb, 
                                int                                 m, 
                                int                                 n, 
                                int                                 k, 
                                const float                        *alpha, 
                                const float                        *A, 
                                int                                 lda, 
                                const float                        *B, 
                                int                                 ldb, 
                                const float                        *beta, 
                                float                              *C, 
                                int                                 ldc );