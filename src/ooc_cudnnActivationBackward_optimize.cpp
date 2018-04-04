

#include "ooc_cudnn.h"

#include <sys/time.h>
#include <float.h>


double cudnnActivationBackward_weight;
double cudnnActivationBackward_bias;
extern double cudaMemcpy_weight;
extern double cudaMemcpy_bias;

cudnnStatus_t CUDNNWINAPI ooc_cudnnActivationBackwardProfile(){

        int i, loop = 1000;
        struct timeval t1, t2;
        double time1, time2;
        double size1, size2;

        size_t n = 1, c = 32, h = 512, w = 512;


        float *x, *dx, *y, *dy;
        float alpha = 1.0, beta = 0.0;
        cudaMalloc((void**)&x, n*c*h*w*sizeof(float));
        cudaMalloc((void**)&dx, n*c*h*w*sizeof(float));
        cudaMalloc((void**)&y, n*c*h*w*sizeof(float));
        cudaMalloc((void**)&dy, n*c*h*w*sizeof(float));

    // profile Activation
        cudnnHandle_t handle;
        cudnnTensorDescriptor_t xDesc, dxDesc, yDesc, dyDesc;
        cudnnActivationDescriptor_t activationDesc;

        cudnnCreate(&handle);
        cudnnCreateTensorDescriptor(&xDesc);
        cudnnCreateTensorDescriptor(&dxDesc);
        cudnnCreateTensorDescriptor(&yDesc);
        cudnnCreateTensorDescriptor(&dyDesc);
        cudnnCreateActivationDescriptor(&activationDesc);

        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN , 0.0);

        // time1, size1
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudnnActivationBackward(handle, activationDesc, &alpha, yDesc, dy, dyDesc, dy, xDesc, x, &beta, dxDesc, dx);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time1 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size1 = (double)(n * c * h * w * sizeof(float));

        // time2, size2
        h /= 2;
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudnnActivationBackward(handle, activationDesc, &alpha, yDesc, dy, dyDesc, dy, xDesc, x, &beta, dxDesc, dx);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time2 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size2 = (double)(n * c * h * w * sizeof(float));


        // cudnnActivationBackward_weight, cudnnActivationBackward_bias
        cudnnActivationBackward_weight = (time1 - time2) / (size1 - size2);
        cudnnActivationBackward_bias = time1 - cudnnActivationBackward_weight * size1;


        cudnnDestroyActivationDescriptor(activationDesc);
        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyTensorDescriptor(dxDesc);
        cudnnDestroyTensorDescriptor(yDesc);
        cudnnDestroyTensorDescriptor(dyDesc);
        cudnnDestroy(handle);

        cudaFree(x); cudaFree(dx); cudaFree(y); cudaFree(dy);


        return CUDNN_STATUS_SUCCESS;
}



double model_activation_back(
                                const size_t                        n,
                                const size_t                        c,
                                const size_t                        h,
                                const size_t                        w,
                                const size_t                        d_x_s,
                                const size_t                        d_dx_s,
                                const size_t                        d_y_s,
                                const size_t                        d_dy_s,
                                const size_t                        data_size,
                                const size_t                        d_mem_free,
                                const int                           n_step,
                                const int                           c_step,
                                const int                           h_step ){

        double t = DBL_MAX;


        size_t d_n = n / n_step; if(n % n_step != 0) d_n++;
        size_t d_c = c / c_step; if(c % c_step != 0) d_c++;
        size_t d_h = h / h_step; if(h % h_step != 0) d_h++;
        
        //check parameter
        if(d_n*d_c*d_h*w >= 2.0*1024.0*1024.0*1024.0) return t;

        size_t d_mem_x = d_n * d_c * d_h * w * data_size * d_x_s;
        size_t d_mem_dx = d_n * d_c * d_h * w * data_size * d_dx_s;
        size_t d_mem_y = d_n * d_c * d_h * w * data_size * d_y_s;
        size_t d_mem_dy = d_n * d_c * d_h * w * data_size * d_dy_s;


        size_t comp_activation = d_n * d_c * d_h * w * data_size;

        if(d_mem_x + d_mem_dx + d_mem_y + d_mem_dy < d_mem_free / 3){
                double t_pre = cudaMemcpy_weight * (d_mem_x + d_mem_y + d_mem_dy) + cudaMemcpy_bias * (d_x_s + d_y_s + d_dy_s);
                double t_activation = cudnnActivationBackward_weight * comp_activation + cudnnActivationBackward_bias;
                double t_off = cudaMemcpy_weight * d_mem_dx + cudaMemcpy_bias * d_dx_s;

                double t_tmp = t_pre;
                if(t_tmp < t_activation) t_tmp = t_activation;
                if(t_tmp < t_off) t_tmp = t_off;

                t = t_pre + t_activation + t_off + (n_step * c_step * h_step - 1) * t_tmp;
        }

        return t;
}


cudnnStatus_t CUDNNWINAPI ooc_cudnnActivationBackward_optimize(
                                const size_t                        n,
                                const size_t                        c,
                                const size_t                        h,
                                const size_t                        w,
                                const bool                          x_on_device,
                                const bool                          dx_on_device,
                                const bool                          y_on_device,
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

        size_t d_x_s = 0; if(!x_on_device) d_x_s = 1;
        size_t d_dx_s = 0; if(!dx_on_device) d_dx_s = 1;
        size_t d_y_s = 0; if(!y_on_device) d_y_s = 1;
        size_t d_dy_s = 0; if(!dy_on_device) d_dy_s = 1;

        int n_step_opt = 1, c_step_opt = 1, h_step_opt = 1;

        tmp = 1;
        for(n_step_opt = 1; n_step_opt <= n; n_step_opt++){
                time = model_activation_back(n, c, h, w, d_x_s, d_dx_s, d_y_s, d_dy_s, data_size, d_mem_free, n_step_opt, c_step_opt, h_step_opt);

                if(time < time_opt){
                        time_opt = time;
                        tmp = n_step_opt;
                }
        }
        n_step_opt = tmp;

        tmp = 1;
        for(h_step_opt = 1; h_step_opt <= h; h_step_opt++){
                time = model_activation_back(n, c, h, w, d_x_s, d_dx_s, d_y_s, d_dy_s, data_size, d_mem_free, n_step_opt, c_step_opt, h_step_opt);
                
                if(time < time_opt){
                        time_opt = time;
                        tmp = h_step_opt;
                }
        }
        h_step_opt = tmp;

        tmp = 1;
        for(c_step_opt = 1; c_step_opt <= c; c_step_opt++){
                time = model_activation_back(n, c, h, w, d_x_s, d_dx_s, d_y_s, d_dy_s, data_size, d_mem_free, n_step_opt, c_step_opt, h_step_opt);
                
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