

#include "ooc_cudnn.h"

#include <sys/time.h>
#include <float.h>


double cudnnSoftmaxForward_weight;
double cudnnSoftmaxForward_bias;
extern double cudaMemcpy_weight;
extern double cudaMemcpy_bias;

cudnnStatus_t CUDNNWINAPI ooc_cudnnSoftmaxForwardProfile(){

        int i, loop = 100;
        struct timeval t1, t2;
        double time1, time2;
        double size1, size2;

        size_t n = 1, c = 128, h = 128, w = 128;

        float *x, *y;
        cudaMalloc((void**)&x, n*c*h*w*sizeof(float));
        cudaMalloc((void**)&y, n*c*h*w*sizeof(float));

    // profile Softmax
        cudnnHandle_t handle;
        cudnnTensorDescriptor_t xDesc, yDesc;

        cudnnCreate(&handle);
        cudnnCreateTensorDescriptor(&xDesc);
        cudnnCreateTensorDescriptor(&yDesc);

        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        float alpha = 1.0;
        float beta = 0.0;

        // time1, size1
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, xDesc, x, &beta, yDesc, y);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time1 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size1 = (double)(n * c * h * w * sizeof(float));

        // time2, size2
        c /= 2;
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for(i = 0; i < loop; i++){
                cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, xDesc, x, &beta, yDesc, y);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time2 = ((double)(t2.tv_usec-t1.tv_usec) * 1000 + (double)(t2.tv_sec-t1.tv_sec) * 1000 * 1000 * 1000) / (double)loop;
        size1 = (double)(n * c * h * w * sizeof(float));


        // cudnnSoftmaxForward_weight, cudnnSoftmaxForward_bias
        cudnnSoftmaxForward_weight = (time1 - time2) / (size1 - size2);
        cudnnSoftmaxForward_bias = time1 - cudnnSoftmaxForward_weight * size1;


        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyTensorDescriptor(yDesc);
        cudnnDestroy(handle);

        cudaFree(x);
        cudaFree(y);


        return CUDNN_STATUS_SUCCESS;
}


double model_softmax(
                                const size_t                        n,
                                const size_t                        c,
                                const size_t                        h,
                                const size_t                        w,
                                const size_t                        d_x_s,
                                const size_t                        d_y_s,
                                const size_t                        data_size,
                                const size_t                        d_mem_free,
                                const int                           n_step,
                                const int                           h_step ){

        double t = DBL_MAX;

        size_t d_n = n / n_step; if(n % n_step != 0) d_n++;
        size_t d_h = h / h_step; if(h % h_step != 0) d_h++;

        //check parameter
        if(d_n*c*d_h*w >= 2.0*1024.0*1024.0*1024.0) return t;

        size_t d_mem_x = d_n * c * d_h * w * data_size * d_x_s;
        size_t d_mem_y = d_n * c * d_h * w * data_size * d_y_s;

        size_t comp_softmax = d_n * c * d_h * w * data_size;

        if(d_mem_x + d_mem_y < d_mem_free){
                double t_pre = cudaMemcpy_weight * d_mem_x + cudaMemcpy_bias * d_x_s;
                double t_softmax = cudnnSoftmaxForward_weight * comp_softmax + cudnnSoftmaxForward_bias;
                double t_off = cudaMemcpy_weight * d_mem_y + cudaMemcpy_bias * d_y_s;

                double t_tmp = t_pre;
                if(t_tmp < t_softmax) t_tmp = t_softmax;
                if(t_tmp < t_off) t_tmp = t_off;

                t = t_pre + t_softmax + t_off + (n_step * h_step - 1) * t_tmp;
        }

        return t;
}



cudnnStatus_t CUDNNWINAPI ooc_cudnnSoftmaxForward_optimize(
                                const size_t                        n,
                                const size_t                        c,
                                const size_t                        h,
                                const size_t                        w,
                                const bool                          x_on_device,
                                const bool                          y_on_device, 
                                const size_t                        data_size,
                                cudnnSoftmaxMode_t                  mode,
                                int                                *n_step,
                                int                                *h_step ){

        int tmp;

        double time;
        double time_opt = DBL_MAX;

        // memory information
        size_t d_mem_free, d_mem_total;
        cudaMemGetInfo(&d_mem_free, &d_mem_total);

        size_t d_x_s = 0; if(!x_on_device) d_x_s = 1;
        size_t d_y_s = 0; if(!y_on_device) d_y_s = 1;

        int n_step_opt = 1, h_step_opt = 1;


        if(mode == CUDNN_SOFTMAX_MODE_CHANNEL){
                tmp = 1;
                for(h_step_opt = 1; h_step_opt <= h; h_step_opt++){
                        time = model_softmax(n, c, h, w, d_x_s, d_y_s, data_size, d_mem_free, n_step_opt, h_step_opt);
                        if(time < time_opt){ 
                                time_opt = time;
                                tmp = h_step_opt;
                        }
                }
                h_step_opt = tmp;
        }


        tmp = 1;
        for(n_step_opt = 1; n_step_opt <= n; n_step_opt++){
                time = model_softmax(n, c, h, w, d_x_s, d_y_s, data_size, d_mem_free, n_step_opt, h_step_opt);
                if(time < time_opt){ 
                        time_opt = time;
                        tmp = n_step_opt;
                }
        }
        n_step_opt = tmp;


        *n_step = n_step_opt;
        *h_step = h_step_opt;

        return CUDNN_STATUS_SUCCESS;
}
