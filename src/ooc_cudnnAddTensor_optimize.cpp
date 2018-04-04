

#include "ooc_cudnn.h"

#include <sys/time.h>
#include <float.h>


double cudnnAddTensor_weight;
double cudnnAddTensor_bias;
extern double cudaMemcpy_weight;
extern double cudaMemcpy_bias;

cudnnStatus_t CUDNNWINAPI ooc_cudnnAddTensorProfile(){

        int i, loop = 1000;
        struct timeval t1, t2;
        double time1, time2;
        double size1, size2;

        unsigned long long int n = 1, c = 64, h = 256, w = 256;

        float *A, *C;
        cudaMalloc((void**)&A, 1*c*1*1*sizeof(float));
        cudaMalloc((void**)&C, n*c*h*w*sizeof(float));

    // profile AddTensor
        cudnnHandle_t handle;
        cudnnTensorDescriptor_t ADesc, CDesc;

        cudnnCreate(&handle);
        cudnnCreateTensorDescriptor(&ADesc);
        cudnnCreateTensorDescriptor(&CDesc);

        cudnnSetTensor4dDescriptor(ADesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c, 1, 1);
        cudnnSetTensor4dDescriptor(CDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        float alpha = 1.0;
        float beta = 1.0;

        // time1, size1
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudnnAddTensor(handle, &alpha, ADesc, A, &beta, CDesc, C);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time1 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size1 = (double)(n * c * h * w * sizeof(float));

        // time2, size2
        h /= 2; w /= 2;
        cudnnSetTensor4dDescriptor(CDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudnnAddTensor(handle, &alpha, ADesc, A, &beta, CDesc, C);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time2 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size1 = (double)(n * c * h * w * sizeof(float));


        // cudnnAddTensor_weight, cudnnAddTensor_bias
        cudnnAddTensor_weight = (time1 - time2) / (size1 - size2);
        cudnnAddTensor_bias = time1 - cudnnAddTensor_weight * size1;

        cudnnDestroyTensorDescriptor(ADesc);
        cudnnDestroyTensorDescriptor(CDesc);
        cudnnDestroy(handle);

        cudaFree(A);
        cudaFree(C);


        return CUDNN_STATUS_SUCCESS;
}


double model_addbias(
                                const size_t                        n,
                                const size_t                        c,
                                const size_t                        h,
                                const size_t                        w,
                                const size_t                        d_A_s,
                                const size_t                        d_C_s,
                                const size_t                        data_size,
                                const size_t                        d_mem_free,
                                int                                 n_step,
                                int                                 c_step,
                                int                                 h_step ){

        double t = DBL_MAX;

        size_t d_n = n / n_step; if(n % n_step != 0) d_n++;
        size_t d_c = c / c_step; if(c % c_step != 0) d_c++;
        size_t d_h = h / h_step; if(h % h_step != 0) d_h++;

        //check parameter
        if((d_n*d_c*d_h*w >= 2.0*1024.0*1024.0*1024.0) || (1*d_c*1*1 >= 2.0*1024.0*1024.0*1024.0)) return t;

        size_t d_mem_A = 1 * d_c * 1 * 1 * data_size * d_A_s;
        size_t d_mem_C = d_n * d_c * d_h * w * data_size * d_C_s;

        if(d_c % 32 != 0) d_c = 32 * (d_c /32 + 1);
        size_t comp_addbias = d_n * d_c * d_h * w * data_size;

        if(d_mem_A + d_mem_C < d_mem_free){
                double t_pre = cudaMemcpy_weight * (d_mem_A + d_mem_C) + cudaMemcpy_bias * (d_A_s + d_C_s);
                double t_addbias = cudnnAddTensor_weight * comp_addbias + cudnnAddTensor_bias;
                double t_off = cudaMemcpy_weight * d_mem_C + cudaMemcpy_bias * d_C_s;

                double t_tmp = t_pre;
                if(t_tmp < t_addbias) t_tmp = t_addbias;
                if(t_tmp < t_off) t_tmp = t_off;

                t = t_pre + t_addbias + t_off + (n_step * c_step * h_step - 1) * t_tmp;
        }

        return t;
}



cudnnStatus_t CUDNNWINAPI ooc_cudnnAddTensor_optimize(
                                const size_t                        n,
                                const size_t                        c,
                                const size_t                        h,
                                const size_t                        w,
                                const bool                          A_on_device,
                                const bool                          C_on_device, 
                                const size_t                        data_size,
                                int                                *n_step,
                                int                                *c_step,
                                int                                *h_step ){


        int tmp;

        double time;
        double time_opt = DBL_MAX;

        // memory information
        size_t d_mem_free, d_mem_total;
        cudaMemGetInfo(&d_mem_free, &d_mem_total);

        size_t d_A_s = 0; if(!A_on_device) d_A_s = 1;
        size_t d_C_s = 0; if(!C_on_device) d_C_s = 1;

        int n_step_opt = 1, c_step_opt = 1, h_step_opt = 1;

        tmp = 1;
        for(n_step_opt = 1; n_step_opt <= n; n_step_opt++){
                time = model_addbias(n, c, h, w, d_A_s, d_C_s, data_size, d_mem_free, n_step_opt, c_step_opt, h_step_opt);
                if(time < time_opt){ 
                        time_opt = time;
                        tmp = n_step_opt;
                }
        }
        n_step_opt = tmp;

        tmp = 1;
        for(h_step_opt = 1; h_step_opt <= h; h_step_opt++){
                time = model_addbias(n, c, h, w, d_A_s, d_C_s, data_size, d_mem_free, n_step_opt, c_step_opt, h_step_opt);
                if(time < time_opt){ 
                        time_opt = time;
                        tmp = h_step_opt;
                }
        }
        h_step_opt = tmp;

        tmp = 1;
        for(c_step_opt = 1; c_step_opt <= c; c_step_opt++){
                time = model_addbias(n, c, h, w, d_A_s, d_C_s, data_size, d_mem_free, n_step_opt, c_step_opt, h_step_opt);
                if(time < time_opt){ 
                        time_opt = time;
                        tmp = c_step_opt;
                }
        }
        c_step_opt = tmp;


        *n_step = n_step_opt;
        *c_step = c_step_opt;
        *h_step = h_step_opt;

        return CUDNN_STATUS_SUCCESS;
}
