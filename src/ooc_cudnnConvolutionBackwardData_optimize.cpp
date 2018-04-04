

#include "ooc_cudnn.h"

#include <sys/time.h>
#include <float.h>


double cudnnConvolutionBackwardData_weight;
double cudnnConvolutionBackwardData_bias;
extern double cudaMemcpy_weight;
extern double cudaMemcpy_bias;

cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBackwardDataProfile(){

        int i, loop = 1000;
        struct timeval t1, t2;
        double time1, time2;
        double size1, size2;

        size_t n = 1, x_c = 128, x_h = 128, x_w = 128, y_c = 128, y_h, y_w, w_h = 3, w_w = 3;

        y_h = (x_h - w_h) + 1;
        y_w = (x_w - w_w) + 1;

        float *dx, *w, *dy, *workspace;
        float alpha = 1.0, beta = 0.0;
        cudaMalloc((void**)&dx, n*x_c*x_h*x_w*sizeof(float));
        cudaMalloc((void**)&w, y_c*x_c*w_h*w_w*sizeof(float));
        cudaMalloc((void**)&dy, n*y_c*y_h*y_w*sizeof(float));

    // profile convolution
        cudnnHandle_t handle;
        cudnnTensorDescriptor_t dxDesc, dyDesc;
        cudnnFilterDescriptor_t wDesc;
        cudnnConvolutionDescriptor_t convDesc;

        cudnnCreate(&handle);
        cudnnCreateTensorDescriptor(&dxDesc);
        cudnnCreateTensorDescriptor(&dyDesc);
        cudnnCreateFilterDescriptor(&wDesc);
        cudnnCreateConvolutionDescriptor(&convDesc);

        cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, x_c, x_h, x_w);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, y_c, y_h, y_w);
        cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_c, x_c, w_h, w_w);
        cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

        // time1, size1
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudnnConvolutionBackwardData(handle, &alpha, wDesc, w, dyDesc, dy, convDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, workspace, 0, &beta, dxDesc, dx);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time1 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size1 = (double)(n * w_h * w_w * x_c * y_h * y_w * y_c * sizeof(float));

        // time2, size2
        x_h /= 2; x_w /= 2; y_h = (x_h - w_h) + 1; y_w = (x_w - w_w) + 1;
        cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, x_c, x_h, x_w);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, y_c, y_h, y_w);

        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudnnConvolutionBackwardData(handle, &alpha, wDesc, w, dyDesc, dy, convDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, workspace, 0, &beta, dxDesc, dx);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time2 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size2 = (double)(n * w_h * w_w * x_c * y_h * y_w * y_c * sizeof(float));


        // cudnnConvolutionBackwardData_weight, cudnnConvolutionBackwardData_bias
        cudnnConvolutionBackwardData_weight = (time1 - time2) / (size1 - size2);
        cudnnConvolutionBackwardData_bias = time1 - cudnnConvolutionBackwardData_weight * size1;


        cudnnDestroyConvolutionDescriptor(convDesc);
        cudnnDestroyFilterDescriptor(wDesc);
        cudnnDestroyTensorDescriptor(dxDesc);
        cudnnDestroyTensorDescriptor(dyDesc);
        cudnnDestroy(handle);

        cudaFree(dx); cudaFree(w); cudaFree(dy);


        return CUDNN_STATUS_SUCCESS;
}



double model_convbackdata(
                                const size_t                        n,
                                const size_t                        x_c,
                                const size_t                        x_h,
                                const size_t                        x_w,
                                const size_t                        w_h,
                                const size_t                        w_w,
                                const size_t                        y_c,
                                const size_t                        y_h,
                                const size_t                        y_w,
                                const size_t                        d_dx_s,
                                const size_t                        d_w_s,
                                const size_t                        d_dy_s,
                                const size_t                        stride_h,
                                const size_t                        dilation_h,
                                const size_t                        data_size,
                                const size_t                        d_mem_free,
                                const int                           n_step,
                                const int                           x_c_step,
                                const int                           y_c_step,
                                const int                           h_step ){

        double t = DBL_MAX;
    
        size_t dilation_w_h = (w_h - 1) * dilation_h + 1;

        size_t d_n2 = n / n_step;
        size_t d_y_c2 = y_c / y_c_step;
        size_t d_x_c2 = x_c / x_c_step;
        size_t d_x_h2 = x_h / h_step;
        size_t d_y_h2 = (d_x_h2 + dilation_w_h - 1) / stride_h;
        if(d_y_h2 < 1) return t;

        size_t d_n = d_n2; if(n % n_step != 0) d_n++;
        size_t d_y_c = d_y_c2; if(y_c % y_c_step != 0) d_y_c++;
        size_t d_x_c = d_x_c2; if(x_c % x_c_step != 0) d_x_c++;
        size_t d_x_h = d_x_h2; if(x_h % h_step != 0) d_x_h++;
        size_t d_y_h = (d_x_h2 + dilation_w_h - 1) / stride_h;
        if(d_y_h < 1) return t;

        //check parameter
        if((d_n*d_x_c*d_x_h*x_w >= 2.0*1024.0*1024.0*1024.0) || (d_n*d_y_c*d_y_h*y_w >= 2.0*1024.0*1024.0*1024.0) || (d_y_c*d_x_c*w_h*w_w >= 2.0*1024.0*1024.0*1024.0))return t;

        size_t d_mem_dx = d_n * d_x_c * d_x_h * x_w * data_size * d_dx_s;
        size_t d_mem_dx2 = d_n2 * d_x_c2 * d_x_h2 * x_w * data_size * d_dx_s;
        size_t d_mem_w = d_y_c * d_x_c * w_h * w_w * data_size * d_w_s;
        size_t d_mem_dy = d_n * d_y_c * d_y_h * y_w * data_size * d_dy_s;

        if(d_y_c % 32 != 0) d_y_c = 32 * (d_y_c / 32 + 1);
        size_t comp_conv = d_n * d_x_c * (d_x_h / stride_h) * y_w * d_y_c * w_h * w_w * data_size;
        if(d_y_c2 % 32 != 0) d_y_c2 = 32 * (d_y_c2 / 32 + 1);
        size_t comp_conv2 = d_n2 * d_x_c2 * (d_x_h2 / stride_h) * y_w * d_y_c2 * w_h * w_w * data_size;

        if(d_mem_dx + d_mem_w + d_mem_dy < d_mem_free / 3){
                double t_pre = cudaMemcpy_weight * (d_mem_dy + d_mem_w) + cudaMemcpy_bias * (d_dy_s + d_w_s);
                double t_conv = cudnnConvolutionBackwardData_weight * comp_conv + cudnnConvolutionBackwardData_bias;
                double t_conv2 = cudnnConvolutionBackwardData_weight * comp_conv2 + cudnnConvolutionBackwardData_bias;
                double t_off = cudaMemcpy_weight * d_mem_dx + cudaMemcpy_bias * d_dx_s;
                double t_off2 = cudaMemcpy_weight * d_mem_dx2 + cudaMemcpy_bias * d_dx_s;

                double t_tmp = t_pre;
                if(t_tmp < t_conv) t_tmp = t_conv;

                double t_tmp2 = t_pre + t_conv + (y_c_step - 1) * t_tmp;
                if(t_tmp2 < t_off){
                        t = t_tmp2 + (n_step * x_c_step * h_step) * t_off;
                }
                else if(t_pre < t_conv){
                        t = t_pre + (n_step * y_c_step * x_c_step * h_step) * t_conv + t_off2;
                }
                else{
                        t = (n_step * y_c_step * x_c_step * h_step) * t_pre + t_conv2 + t_off2;
                }
        }

        return t;
}


cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBackwardData_optimize(
                                const size_t                        n,
                                const size_t                        x_c,
                                const size_t                        x_h,
                                const size_t                        x_w,
                                const size_t                        w_h,
                                const size_t                        w_w,
                                const size_t                        y_c,
                                const size_t                        y_h,
                                const size_t                        y_w,
                                const bool                          dx_on_device,
                                const bool                          w_on_device,
                                const bool                          dy_on_device,
                                const size_t                        stride_h,
                                const size_t                        dilation_h,
                                const size_t                        data_size,
                                int                                *n_step,
                                int                                *x_c_step,
                                int                                *y_c_step,
                                int                                *h_step ){

        int tmp;

        double time;
        double time_opt = DBL_MAX;

        // memory information
        size_t d_mem_free, d_mem_total;
        cudaMemGetInfo(&d_mem_free, &d_mem_total);

        size_t d_dx_s = 0; if(!dx_on_device) d_dx_s = 1;
        size_t d_w_s = 0; if(!w_on_device) d_w_s = 1;
        size_t d_dy_s = 0; if(!dy_on_device) d_dy_s = 1;

        int n_step_opt = 1, y_c_step_opt = 1, x_c_step_opt = 1, h_step_opt = 1;

        tmp = 1;
        for(n_step_opt = 1; n_step_opt <= n; n_step_opt++){
                time = model_convbackdata(n, x_c, x_h, x_w, w_h, w_w, y_c, y_h, y_w, d_dx_s, d_w_s, d_dy_s, 
                                   stride_h, dilation_h, data_size, d_mem_free, n_step_opt, x_c_step_opt, y_c_step_opt, h_step_opt);
                if(time < time_opt){
                        time_opt = time;
                        tmp = n_step_opt;
                }
        }
        if(time_opt > DBL_MAX / 2.0) tmp = n;
        n_step_opt = tmp;

        tmp = 1;
        for(h_step_opt = 1; h_step_opt <= x_h; h_step_opt++){
                time = model_convbackdata(n, x_c, x_h, x_w, w_h, w_w, y_c, y_h, y_w, d_dx_s, d_w_s, d_dy_s, 
                                   stride_h, dilation_h, data_size, d_mem_free, n_step_opt, x_c_step_opt, y_c_step_opt, h_step_opt);
                if(time < time_opt){
                        time_opt = time;
                        tmp = h_step_opt;
                }
        }
        if(time_opt > DBL_MAX / 2.0) tmp = x_h;
        h_step_opt = tmp;

        tmp = 1;
        for(x_c_step_opt = 1; x_c_step_opt <= x_c; x_c_step_opt++){
                time = model_convbackdata(n, x_c, x_h, x_w, w_h, w_w, y_c, y_h, y_w, d_dx_s, d_w_s, d_dy_s, 
                                   stride_h, dilation_h, data_size, d_mem_free, n_step_opt, x_c_step_opt, y_c_step_opt, h_step_opt);
                if(time < time_opt){
                        time_opt = time;
                        tmp = x_c_step_opt;
                }
        }
        if(time_opt > DBL_MAX / 2.0) tmp = x_c;
        x_c_step_opt = tmp;

        tmp = 1;
        for(y_c_step_opt = 1; y_c_step_opt <= y_c; y_c_step_opt++){
                time = model_convbackdata(n, x_c, x_h, x_w, w_h, w_w, y_c, y_h, y_w, d_dx_s, d_w_s, d_dy_s, 
                                   stride_h, dilation_h, data_size, d_mem_free, n_step_opt, x_c_step_opt, y_c_step_opt, h_step_opt);
                if(time < time_opt){
                        time_opt = time;
                        tmp = y_c_step_opt;
                }
        }
        if(time_opt > DBL_MAX / 2.0) tmp = y_c;
        y_c_step_opt = tmp;


        *n_step = n_step_opt;
        *x_c_step = x_c_step_opt;
        *y_c_step = y_c_step_opt;
        *h_step = h_step_opt;

        return CUDNN_STATUS_SUCCESS;
}