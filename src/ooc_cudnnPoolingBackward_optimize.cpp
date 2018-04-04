

#include "ooc_cudnn.h"

#include <sys/time.h>
#include <float.h>


double cudnnPoolingBackward_weight;
double cudnnPoolingBackward_bias;
extern double cudaMemcpy_weight;
extern double cudaMemcpy_bias;

cudnnStatus_t CUDNNWINAPI ooc_cudnnPoolingBackwardProfile(){

        int i, loop = 1000;
        struct timeval t1, t2;
        double time1, time2;
        double size1, size2;

        size_t n = 1, c = 64, x_h = 256, x_w = 256, y_h, y_w, window_h = 2, window_w = 2;
        size_t stride_h = 2, stride_w = 2;

        y_h = (x_h - window_h) / stride_h + 1;
        y_w = (x_w - window_w) / stride_w + 1;

        float *x, *dx, *y, *dy;
        float alpha = 1.0, beta = 0.0;
        cudaMalloc((void**)&x, n*c*x_h*x_w*sizeof(float));
        cudaMalloc((void**)&dx, n*c*x_h*x_w*sizeof(float));
        cudaMalloc((void**)&y, n*c*y_h*y_w*sizeof(float));
        cudaMalloc((void**)&dy, n*c*y_h*y_w*sizeof(float));

    // profile Pooling
        cudnnHandle_t handle;
        cudnnTensorDescriptor_t xDesc, dxDesc, yDesc, dyDesc;
        cudnnPoolingDescriptor_t poolingDesc;

        cudnnCreate(&handle);
        cudnnCreateTensorDescriptor(&xDesc);
        cudnnCreateTensorDescriptor(&dxDesc);
        cudnnCreateTensorDescriptor(&yDesc);
        cudnnCreateTensorDescriptor(&dyDesc);
        cudnnCreatePoolingDescriptor(&poolingDesc);

        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, x_h, x_w);
        cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, x_h, x_w);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, y_h, y_w);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, y_h, y_w);
        cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, window_h, window_w, 0, 0, stride_h, stride_w);

        // time1, size1
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudnnPoolingBackward(handle, poolingDesc, &alpha, yDesc, dy, yDesc, dy, xDesc, x, &beta, dxDesc, dx);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time1 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size1 = (double)(n * c * y_h * y_w * window_h * window_w * sizeof(float));

        // time2, size2
        x_h /= 2; y_h = (x_h - window_h) / stride_h + 1;
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, x_h, x_w);
        cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, x_h, x_w);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, y_h, y_w);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, y_h, y_w);

        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudnnPoolingBackward(handle, poolingDesc, &alpha, yDesc, dy, yDesc, dy, xDesc, x, &beta, dxDesc, dx);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time2 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size2 = (double)(n * c * y_h * y_w * window_h * window_w * sizeof(float));


        // cudnnPoolingBackward_weight, cudnnPoolingBackward_bias
        cudnnPoolingBackward_weight = (time1 - time2) / (size1 - size2);
        cudnnPoolingBackward_bias = time1 - cudnnPoolingBackward_weight * size1;

        cudnnDestroyPoolingDescriptor(poolingDesc);
        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyTensorDescriptor(dxDesc);
        cudnnDestroyTensorDescriptor(yDesc);
        cudnnDestroyTensorDescriptor(dyDesc);
        cudnnDestroy(handle);

        cudaFree(x); cudaFree(dx); cudaFree(y); cudaFree(dy);


        return CUDNN_STATUS_SUCCESS;
}



double model_pooling_back(
                                const size_t                        n,
                                const size_t                        c,
                                const size_t                        x_h,
                                const size_t                        x_w,
                                const size_t                        window_h,
                                const size_t                        window_w,
                                const size_t                        y_h,
                                const size_t                        y_w,
                                const size_t                        d_x_s,
                                const size_t                        d_dx_s,
                                const size_t                        d_y_s,
                                const size_t                        d_dy_s,
                                const size_t                        stride_h,
                                const size_t                        data_size,
                                const size_t                        d_mem_free,
                                const int                           n_step,
                                const int                           c_step,
                                const int                           h_step ){

        double t = DBL_MAX;


        size_t d_n = n / n_step; if(n % n_step != 0) d_n++;
        size_t d_c = c / c_step; if(c % c_step != 0) d_c++;
        size_t d_x_h = x_h / h_step; if(x_h % h_step != 0) d_x_h++;
        size_t d_y_h = (d_x_h + window_h - 1) / stride_h;
        if(d_y_h < 1) return t;

        //check parameter
        if((d_n*d_c*d_x_h*x_w >= 2.0*1024.0*1024.0*1024.0) || (d_n*d_c*d_y_h*y_w >= 2.0*1024.0*1024.0*1024.0))return t;

        size_t d_mem_x = d_n * d_c * d_x_h * x_w * data_size * d_x_s;
        size_t d_mem_dx = d_n * d_c * d_x_h * x_w * data_size * d_dx_s;
        size_t d_mem_y = d_n * d_c * d_y_h * y_w * data_size * d_y_s;
        size_t d_mem_dy = d_n * d_c * d_y_h * y_w * data_size * d_dy_s;

        size_t d_mem_x_buffer = d_mem_x;
        if(h_step != 1) d_mem_x_buffer = d_n * d_c * d_x_h * x_w * data_size;
        size_t d_mem_y_buffer = d_mem_y;
        if(h_step != 1) d_mem_y_buffer = d_n * d_c * d_y_h * y_w * data_size;
        size_t d_mem_dy_buffer = d_mem_dy;
        if(h_step != 1) d_mem_dy_buffer = d_n * d_c * d_y_h * y_w * data_size;

        size_t comp_pooling = d_n * d_c * (d_x_h / stride_h) * y_w * window_h * window_w * data_size;

        if(d_mem_x_buffer + d_mem_dx + d_mem_y_buffer + d_mem_dy_buffer < d_mem_free / 3){
                double t_pre = cudaMemcpy_weight * (d_mem_x + d_mem_y_buffer + d_mem_dy_buffer) + cudaMemcpy_bias * (d_x_s + d_y_s + d_dy_s);
                double t_pooling = cudnnPoolingBackward_weight * comp_pooling + cudnnPoolingBackward_bias;
                double t_off = cudaMemcpy_weight * d_mem_dx + cudaMemcpy_bias * d_dx_s;

                double t_tmp = t_pre;
                if(t_tmp < t_pooling) t_tmp = t_pooling;
                if(t_tmp < t_off) t_tmp = t_off;

                t = t_pre + t_pooling + t_off + (n_step * c_step * h_step - 1) * t_tmp;
        }

        return t;
}


cudnnStatus_t CUDNNWINAPI ooc_cudnnPoolingBackward_optimize(
                                const size_t                        n,
                                const size_t                        c,
                                const size_t                        x_h,
                                const size_t                        x_w,
                                const size_t                        window_h,
                                const size_t                        window_w,
                                const size_t                        y_h,
                                const size_t                        y_w,
                                const bool                          x_on_device,
                                const bool                          dx_on_device,
                                const bool                          y_on_device,
                                const bool                          dy_on_device,
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
        size_t d_dx_s = 0; if(!dx_on_device) d_dx_s = 1;
        size_t d_y_s = 0; if(!y_on_device) d_y_s = 1;
        size_t d_dy_s = 0; if(!dy_on_device) d_dy_s = 1;

        int n_step_opt = 1, c_step_opt = 1, h_step_opt = 1;

        tmp = 1;
        for(n_step_opt = 1; n_step_opt <= n; n_step_opt++){
                time = model_pooling_back(n, c, x_h, x_w, window_h, window_w, y_h, y_w, d_x_s, d_dx_s, d_y_s, d_dy_s, 
                                            stride_h, data_size, d_mem_free, n_step_opt, c_step_opt, h_step_opt);
                if(time < time_opt){
                        time_opt = time;
                        tmp = n_step_opt;
                }
        }
        n_step_opt = tmp;

        tmp = 1;
        for(h_step_opt = 1; h_step_opt <= y_h; h_step_opt++){
                time = model_pooling_back(n, c, x_h, x_w, window_h, window_w, y_h, y_w, d_x_s, d_dx_s, d_y_s, d_dy_s, 
                                            stride_h, data_size, d_mem_free, n_step_opt, c_step_opt, h_step_opt);
                if(time < time_opt){
                        time_opt = time;
                        tmp = h_step_opt;
                }
        }
        h_step_opt = tmp;

        tmp = 1;
        for(c_step_opt = 1; c_step_opt <= c; c_step_opt++){
                time = model_pooling_back(n, c, x_h, x_w, window_h, window_w, y_h, y_w, d_x_s, d_dx_s, d_y_s, d_dy_s, 
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