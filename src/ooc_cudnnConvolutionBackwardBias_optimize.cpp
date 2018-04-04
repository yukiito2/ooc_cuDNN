

#include "ooc_cudnn.h"

#include <sys/time.h>
#include <float.h>


double cudnnConvolutionBackwardBias_weight;
double cudnnConvolutionBackwardBias_bias;
extern double cudaMemcpy_weight;
extern double cudaMemcpy_bias;

cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBackwardBiasProfile(){

        int i, loop = 1000;
        struct timeval t1, t2;
        double time1, time2;
        double size1, size2;

        unsigned long long int n = 1, c = 256, h = 128, w = 128;

        float *db, *dy;
        cudaMalloc((void**)&db, 1*c*1*1*sizeof(float));
        cudaMalloc((void**)&dy, n*c*h*w*sizeof(float));

    // profile ConvolutionBackwardBias
        cudnnHandle_t handle;
        cudnnTensorDescriptor_t dbDesc, dyDesc;

        cudnnCreate(&handle);
        cudnnCreateTensorDescriptor(&dbDesc);
        cudnnCreateTensorDescriptor(&dyDesc);

        cudnnSetTensor4dDescriptor(dbDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c, 1, 1);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        float alpha = 1.0;
        float beta = 1.0;

        // time1, size1
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudnnConvolutionBackwardBias(handle, &alpha, dyDesc, dy, &beta, dbDesc, db);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time1 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size1 = (double)(n * c * h * w * sizeof(float));

        // time2, size2
        c /= 2;
        cudnnSetTensor4dDescriptor(dbDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c, 1, 1);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudnnConvolutionBackwardBias(handle, &alpha, dyDesc, dy, &beta, dbDesc, db);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time2 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size1 = (double)(n * c * h * w * sizeof(float));


        // cudnnConvolutionBackwardBias_weight, cudnnConvolutionBackwardBias_bias
        cudnnConvolutionBackwardBias_weight = (time1 - time2) / (size1 - size2);
        cudnnConvolutionBackwardBias_bias = time1 - cudnnConvolutionBackwardBias_weight * size1;


        cudnnDestroyTensorDescriptor(dbDesc);
        cudnnDestroyTensorDescriptor(dyDesc);
        cudnnDestroy(handle);

        cudaFree(db);
        cudaFree(dy);


        return CUDNN_STATUS_SUCCESS;
}


double model_convbackbias(
                                const size_t                        n,
                                const size_t                        c,
                                const size_t                        h,
                                const size_t                        w,
                                const size_t                        d_db_s,
                                const size_t                        d_dy_s,
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

        size_t d_mem_db = 1 * d_c * 1 * 1 * data_size * d_db_s;
        size_t d_mem_dy = d_n * d_c * d_h * w * data_size * d_dy_s;

        if(d_c % 32 != 0) d_c = 32 * (d_c /32 + 1);
        size_t comp_convbackbias = d_n * d_c * d_h * w * data_size;

        if(d_mem_db + d_mem_dy < d_mem_free){
                double t_pre = cudaMemcpy_weight * (d_mem_db + d_mem_dy) + cudaMemcpy_bias * (d_db_s + d_dy_s);
                double t_convbackbias = cudnnConvolutionBackwardBias_weight * comp_convbackbias + cudnnConvolutionBackwardBias_bias;
                double t_off = cudaMemcpy_weight * d_mem_db + cudaMemcpy_bias * d_db_s;

                double t_tmp = t_pre;
                if(t_tmp < t_convbackbias) t_tmp = t_convbackbias;
                if(t_tmp < t_off) t_tmp = t_off;

                t = t_pre + t_convbackbias + t_off + (n_step * c_step * h_step - 1) * t_tmp;
        }

        return t;
}



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
                                int                                *h_step ){


        int tmp;

        double time;
        double time_opt = DBL_MAX;

        // memory information
        size_t d_mem_free, d_mem_total;
        cudaMemGetInfo(&d_mem_free, &d_mem_total);

        size_t d_db_s = 0; if(!db_on_device) d_db_s = 1;
        size_t d_dy_s = 0; if(!dy_on_device) d_dy_s = 1;

        int n_step_opt = 1, c_step_opt = 1, h_step_opt = 1;

        tmp = 1;
        for(n_step_opt = 1; n_step_opt <= n; n_step_opt++){
                time = model_convbackbias(n, c, h, w, d_db_s, d_dy_s, data_size, d_mem_free, n_step_opt, c_step_opt, h_step_opt);
                if(time < time_opt){ 
                        time_opt = time;
                        tmp = n_step_opt;
                }
        }
        n_step_opt = tmp;

        tmp = 1;
        for(h_step_opt = 1; h_step_opt <= h; h_step_opt++){
                time = model_convbackbias(n, c, h, w, d_db_s, d_dy_s, data_size, d_mem_free, n_step_opt, c_step_opt, h_step_opt);
                if(time < time_opt){ 
                        time_opt = time;
                        tmp = h_step_opt;
                }
        }
        h_step_opt = tmp;

        tmp = 1;
        for(c_step_opt = 1; c_step_opt <= c; c_step_opt++){
                time = model_convbackbias(n, c, h, w, d_db_s, d_dy_s, data_size, d_mem_free, n_step_opt, c_step_opt, h_step_opt);
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
