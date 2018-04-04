

#include "ooc_cudnn.h"

#include <sys/time.h>
#include <float.h>


double cudnnPoolingForward_weight;
double cudnnPoolingForward_bias;
extern double cudaMemcpy_weight;
extern double cudaMemcpy_bias;

cudnnStatus_t CUDNNWINAPI ooc_cudnnPoolingForwardProfile(){

        int i, loop = 1000;
        struct timeval t1, t2;
        double time1, time2;
        double size1, size2;

        size_t n = 1, c = 64, x_h = 256, x_w = 256, y_h, y_w, window_h = 2, window_w = 2;
        size_t stride_h = 2, stride_w = 2;

        y_h = (x_h - window_h) / stride_h + 1;
        y_w = (x_w - window_w) / stride_w + 1;

        float *x, *y;
        float alpha = 1.0, beta = 0.0;
        cudaMalloc((void**)&x, n*c*x_h*x_w*sizeof(float));
        cudaMalloc((void**)&y, n*c*y_h*y_w*sizeof(float));

    // profile Pooling
        cudnnHandle_t handle;
        cudnnTensorDescriptor_t xDesc, yDesc;
        cudnnPoolingDescriptor_t poolingDesc;

        cudnnCreate(&handle);
        cudnnCreateTensorDescriptor(&xDesc);
        cudnnCreateTensorDescriptor(&yDesc);
        cudnnCreatePoolingDescriptor(&poolingDesc);

        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, x_h, x_w);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, y_h, y_w);
        cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, window_h, window_w, 0, 0, stride_h, stride_w);

        // time1, size1
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudnnPoolingForward(handle, poolingDesc, &alpha, xDesc, x, &beta, yDesc, y);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time1 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size1 = (double)(n * c * y_h * y_w * window_h * window_w * sizeof(float));

        // time2, size2
        x_h /= 2; y_h = (x_h - window_h) / stride_h + 1;
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, x_h, x_w);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, y_h, y_w);

        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudnnPoolingForward(handle, poolingDesc, &alpha, xDesc, x, &beta, yDesc, y);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time2 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size2 = (double)(n * c * y_h * y_w * window_h * window_w * sizeof(float));


        // cudnnPoolingForward_weight, cudnnPoolingForward_bias
        cudnnPoolingForward_weight = (time1 - time2) / (size1 - size2);
        cudnnPoolingForward_bias = time1 - cudnnPoolingForward_weight * size1;

        cudnnDestroyPoolingDescriptor(poolingDesc);
        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyTensorDescriptor(yDesc);
        cudnnDestroy(handle);

        cudaFree(x); cudaFree(y);


        return CUDNN_STATUS_SUCCESS;
}



double model_pooling(
                                const size_t                        n,
                                const size_t                        c,
                                const size_t                        x_h,
                                const size_t                        x_w,
                                const size_t                        window_h,
                                const size_t                        window_w,
                                const size_t                        y_h,
                                const size_t                        y_w,
                                const size_t                        d_x_s,
                                const size_t                        d_y_s,
                                const size_t                        stride_h,
                                const size_t                        data_size,
                                const size_t                        d_mem_free,
                                const int                           n_step,
                                const int                           c_step,
                                const int                           h_step ){

        double t = DBL_MAX;


        size_t d_n = n / n_step; if(n % n_step != 0) d_n++;
        size_t d_c = c / c_step; if(c % c_step != 0) d_c++;
        size_t d_y_h = y_h / h_step; if(y_h % h_step != 0) d_y_h++;
        size_t d_x_h = (d_y_h - 1) * stride_h + window_h;


        //check parameter
        if((d_n*d_c*d_x_h*x_w >= 2.0*1024.0*1024.0*1024.0) || (d_n*d_c*d_y_h*y_w >= 2.0*1024.0*1024.0*1024.0))return t;

        size_t d_mem_x = d_n * d_c * d_x_h * x_w * data_size * d_x_s;
        size_t d_mem_y = d_n * d_c * d_y_h * y_w * data_size * d_y_s;

        size_t d_mem_x_buffer = d_mem_x;
        if(h_step != 1) d_mem_x_buffer = d_n * d_c * d_x_h * x_w * data_size;

        size_t comp_pooling = d_n * d_c * d_y_h * y_w * window_h * window_w * data_size;

        if(d_mem_x_buffer + d_mem_y < d_mem_free / 3){
                double t_pre = cudaMemcpy_weight * d_mem_x + cudaMemcpy_bias * d_x_s;
                double t_pooling = cudnnPoolingForward_weight * comp_pooling + cudnnPoolingForward_bias;
                double t_off = cudaMemcpy_weight * d_mem_y + cudaMemcpy_bias * d_y_s;

                double t_tmp = t_pre;
                if(t_tmp < t_pooling) t_tmp = t_pooling;
                if(t_tmp < t_off) t_tmp = t_off;

                t = t_pre + t_pooling + t_off + (n_step * c_step * h_step - 1) * t_tmp;
        }

        return t;
}


cudnnStatus_t CUDNNWINAPI ooc_cudnnPoolingForward_optimize(
                                const size_t                        n,
                                const size_t                        c,
                                const size_t                        x_h,
                                const size_t                        x_w,
                                const size_t                        window_h,
                                const size_t                        window_w,
                                const size_t                        y_h,
                                const size_t                        y_w,
                                const bool                          x_on_device,
                                const bool                          y_on_device,
                                const size_t                        stride_h,
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

        size_t d_x_s = 0; if(!x_on_device) d_x_s = 1;
        size_t d_y_s = 0; if(!y_on_device) d_y_s = 1;

        int n_step_opt = 1, c_step_opt = 1, h_step_opt = 1;

        tmp = 1;
        for(n_step_opt = 1; n_step_opt <= n; n_step_opt++){
                time = model_pooling(n, c, x_h, x_w, window_h, window_w, y_h, y_w, d_x_s, d_y_s, 
                                            stride_h, data_size, d_mem_free, n_step_opt, c_step_opt, h_step_opt);
                if(time < time_opt){
                        time_opt = time;
                        tmp = n_step_opt;
                }
        }
        n_step_opt = tmp;

        tmp = 1;
        for(h_step_opt = 1; h_step_opt <= y_h; h_step_opt++){
                time = model_pooling(n, c, x_h, x_w, window_h, window_w, y_h, y_w, d_x_s, d_y_s, 
                                            stride_h, data_size, d_mem_free, n_step_opt, c_step_opt, h_step_opt);
                if(time < time_opt){
                        time_opt = time;
                        tmp = h_step_opt;
                }
        }
        h_step_opt = tmp;

        tmp = 1;
        for(c_step_opt = 1; c_step_opt <= c; c_step_opt++){
                time = model_pooling(n, c, x_h, x_w, window_h, window_w, y_h, y_w, d_x_s, d_y_s, 
                                            stride_h, data_size, d_mem_free, n_step_opt, c_step_opt, h_step_opt);
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