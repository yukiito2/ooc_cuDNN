
#include "ooc_cudnn.h"
#include "ooc_cublas.h"

#include <sys/time.h>
#include <float.h>


double cublasSgemm_weight;
double cublasSgemm_bias;
extern double cudaMemcpy_weight;
extern double cudaMemcpy_bias;

cublasStatus_t CUBLASWINAPI ooc_cublasSgemmProfile(){

        int i, loop = 1000;
        struct timeval t1, t2;
        double time1, time2;
        double size1, size2;

        size_t m = 256, n = 256, k = 256;

        float *A, *B, *C;
        float alpha = 1.0, beta = 0.0;
        cudaMalloc((void**)&A, m*k*sizeof(float));
        cudaMalloc((void**)&B, k*n*sizeof(float));
        cudaMalloc((void**)&C, m*n*sizeof(float));

        // profile sgemm
        cublasHandle_t handle;
        cublasCreate(&handle);

        // time1, size1
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, m, B, n, &beta, C, m);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time1 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size1 = (double)(m * n * k * sizeof(float));

        // time2, size2
        m /= 2;

        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, m, B, n, &beta, C, m);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time2 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size2 = (double)(m * n * k * sizeof(float));


        // cublasSgemm_weight, cublasSgemm_bias
        cublasSgemm_weight = (time1 - time2) / (size1 - size2);
        cublasSgemm_bias = time1 - cublasSgemm_weight * size1;

        cudaFree(A); cudaFree(B); cudaFree(C);


        return CUBLAS_STATUS_SUCCESS;
}



double model_sgemm(
                                const size_t                        m,
                                const size_t                        n,
                                const size_t                        k,
                                const size_t                        d_A_s,
                                const size_t                        d_B_s,
                                const size_t                        d_C_s,
                                const size_t                        data_size,
                                const size_t                        d_mem_free,
                                const int                           m_step,
                                const int                           n_step,
                                const int                           k_step ){

        double t = DBL_MAX;

        size_t d_m = m / m_step; if(m % m_step != 0) d_m++;
        size_t d_n = n / n_step; if(n % n_step != 0) d_n++;
        size_t d_k = k / k_step; if(k % k_step != 0) d_k++;

        size_t d_mem_A = d_m * d_k * data_size * d_A_s;
        size_t d_mem_B = d_k * d_n * data_size * d_B_s;
        size_t d_mem_C = d_m * d_n * data_size * d_C_s;

        size_t comp_sgemm = d_m * d_n * d_k * data_size;

        if(d_mem_A + d_mem_B + d_mem_C < d_mem_free / 3){
                double t_pre = cudaMemcpy_weight * (d_mem_A + d_mem_B) + cudaMemcpy_bias * (d_A_s + d_B_s);
                double t_sgemm = cublasSgemm_weight * comp_sgemm + cublasSgemm_bias;
                double t_off = cudaMemcpy_weight * d_mem_C + cudaMemcpy_bias * d_C_s;

                double t_tmp = t_pre;
                if(t_tmp < t_sgemm) t_tmp = t_sgemm;

                double t_tmp2 = k_step * t_tmp;
                if(t_tmp2 < t_off) t_tmp2 = t_off;

                t = t_pre + t_sgemm + t_off + (k_step - 1) * t_tmp + (m_step * n_step - 1) * t_tmp2;
        }

        return t;
}


cublasStatus_t CUBLASWINAPI ooc_cublasSgemm_optimize(
                                const size_t                        m,
                                const size_t                        n,
                                const size_t                        k,
                                const bool                          A_on_device,
                                const bool                          B_on_device,
                                const bool                          C_on_device,
                                const size_t                        data_size,
                                int                                *m_step,
                                int                                *n_step,
                                int                                *k_step ){

        int tmp;

        double time;
        double time_opt = DBL_MAX;

        // memory information
        size_t d_mem_free, d_mem_total;
        cudaMemGetInfo(&d_mem_free, &d_mem_total);

        size_t d_A_s = 0; if(!A_on_device) d_A_s = 1;
        size_t d_B_s = 0; if(!B_on_device) d_B_s = 1;
        size_t d_C_s = 0; if(!C_on_device) d_C_s = 1;

        int m_step_opt = 1, n_step_opt = 1, k_step_opt = 1;

        tmp = 1;
        for(m_step_opt = 1; m_step_opt <= m; m_step_opt++){
                time = model_sgemm(m, n, k, d_A_s, d_B_s, d_C_s, data_size, d_mem_free, m_step_opt, n_step_opt, k_step_opt);
                if(time < time_opt){
                        time_opt = time;
                        tmp = m_step_opt;
                }
        }
        if(time_opt > DBL_MAX / 2.0) tmp = m;
        m_step_opt = tmp;

        tmp = 1;
        for(k_step_opt = 1; k_step_opt <= k; k_step_opt++){
                time = model_sgemm(m, n, k, d_A_s, d_B_s, d_C_s, data_size, d_mem_free, m_step_opt, n_step_opt, k_step_opt);

                if(time < time_opt){
                        time_opt = time;
                        tmp = k_step_opt;
                }
        }
        if(time_opt > DBL_MAX / 2.0) tmp = k;
        k_step_opt = tmp;

        tmp = 1;
        for(n_step_opt = 1; n_step_opt <= n; n_step_opt++){
                time = model_sgemm(m, n, k, d_A_s, d_B_s, d_C_s, data_size, d_mem_free, m_step_opt, n_step_opt, k_step_opt);

                if(time < time_opt){
                        time_opt = time;
                        tmp = n_step_opt;
                }
        }
        if(time_opt > DBL_MAX / 2.0) tmp = n;
        n_step_opt = tmp;
    

        *m_step = m_step_opt;
        *n_step = n_step_opt;
        *k_step = k_step_opt;

        return CUBLAS_STATUS_SUCCESS;
}