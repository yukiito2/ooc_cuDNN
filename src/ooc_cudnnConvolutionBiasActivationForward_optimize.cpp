

#include "ooc_cudnn.h"

#include <sys/time.h>
#include <float.h>


extern double cudnnConvolutionForward_weight;
extern double cudnnConvolutionForward_bias;
extern double cudnnAddTensor_weight;
extern double cudnnAddTensor_bias;
extern double cudnnActivationForward_weight;
extern double cudnnActivationForward_bias;
extern double cudaMemcpy_weight;
extern double cudaMemcpy_bias;


double model_conv_addbias_activ(
                                const size_t                        n,
                                const size_t                        x_c,
                                const size_t                        x_h,
                                const size_t                        x_w,
                                const size_t                        w_h,
                                const size_t                        w_w,
                                const size_t                        y_c,
                                const size_t                        y_h,
                                const size_t                        y_w,
                                const size_t                        d_x_s,
                                const size_t                        d_w_s,
                                const size_t                        d_b_s,
                                const size_t                        d_y_s,
                                const size_t                        stride_h,
                                const size_t                        dilation_h,
                                const size_t                        data_size,
                                const size_t                        d_mem_free,
                                const int                           n_step,
                                const int                           y_c_step,
                                const int                           x_c_step,
                                const int                           h_step ){

        double t = DBL_MAX;
    
        size_t dilation_w_h = (w_h - 1) * dilation_h + 1;

        size_t d_n = n / n_step; if(n % n_step != 0) d_n++;
        size_t d_y_c = y_c / y_c_step; if(y_c % y_c_step != 0) d_y_c++;
        size_t d_x_c = x_c / x_c_step; if(x_c % x_c_step != 0) d_x_c++;
        size_t d_y_h = y_h / h_step; if(y_h % h_step != 0) d_y_h++;
        size_t d_x_h = (d_y_h - 1) * stride_h + dilation_w_h;


        //check parameter
        size_t max_size = (size_t)2 * (size_t)1024 * (size_t)1024 * (size_t)1024;
        if((d_n*d_x_c*d_x_h*x_w >= max_size) || (d_n*d_y_c*d_y_h*y_w >= max_size) || (d_y_c*d_x_c*w_h*w_w >= max_size))return t;

        size_t d_mem_x = d_n * d_x_c * d_x_h * x_w * data_size * d_x_s;
        size_t d_mem_w = d_y_c * d_x_c * w_h * w_w * data_size * d_w_s;
        size_t d_mem_b = 1 * d_y_c * 1 * 1 * data_size * d_b_s;
        size_t d_mem_y = d_n * d_y_c * d_y_h * y_w * data_size * d_y_s;

        size_t d_mem_x_buffer = d_mem_x;
        if(h_step != 1) d_mem_x_buffer = d_n * d_x_c * d_x_h * x_w * data_size;

        if(d_y_c % 32 != 0) d_y_c = 32 * (d_y_c / 32 + 1);
        size_t comp_conv = d_n * d_y_c * d_y_h * y_w * d_x_c * w_h * w_w * data_size;
        size_t comp_addbias = d_n * d_y_c * d_y_h * y_w * data_size;
        size_t comp_activ = d_n * d_y_c * d_y_h * y_w * data_size;

        if(d_mem_x_buffer + d_mem_w + d_mem_y < d_mem_free / 3){
                double t_pre = cudaMemcpy_weight * (d_mem_x + d_mem_w) + cudaMemcpy_bias * (d_x_s + d_w_s);
                double t_conv = cudnnConvolutionForward_weight * comp_conv + cudnnConvolutionForward_bias;
                double t_off =  cudaMemcpy_weight * d_mem_b + cudaMemcpy_bias * d_b_s;
                                + cudnnAddTensor_weight * comp_addbias + cudnnAddTensor_bias
                                + cudnnActivationForward_weight * comp_activ + cudnnActivationForward_bias
                                + cudaMemcpy_weight * d_mem_y + cudaMemcpy_bias * d_y_s;

                double t_tmp1 = t_pre;
                if(t_tmp1 < t_conv) t_tmp1 = t_conv;

                double t_tmp2 = t_pre + t_conv + (x_c_step - 1) * t_tmp1;
                if(t_tmp2 < t_off){
                        t = t_tmp2 + n_step * y_c_step * h_step * t_off;
                }
                else if(t_pre < t_conv){
                        t = t_pre + cudnnConvolutionForward_weight * n * y_c * y_h * y_w * x_c * w_h * w_w * data_size
                            + n_step * y_c_step * x_c_step * h_step * cudnnConvolutionForward_bias + t_off;
                }
                else{
                        t = n_step * y_c_step * x_c_step * h_step * t_pre + t_conv + t_off;
                }
        }

        return t;
}


cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBiasActivationForward_optimize(
                                const size_t                        n,
                                const size_t                        x_c,
                                const size_t                        x_h,
                                const size_t                        x_w,
                                const size_t                        w_h,
                                const size_t                        w_w,
                                const size_t                        y_c,
                                const size_t                        y_h,
                                const size_t                        y_w,
                                const bool                          x_on_device,
                                const bool                          w_on_device,
                                const bool                          b_on_device,
                                const bool                          y_on_device,
                                const size_t                        stride_h,
                                const size_t                        dilation_h,
                                const size_t                        data_size,
                                int                                *n_step,
                                int                                *y_c_step,
                                int                                *x_c_step,
                                int                                *h_step ){

        int tmp;

        double time;
        double time_opt = DBL_MAX;

        // memory information
        size_t d_mem_free, d_mem_total;
        cudaMemGetInfo(&d_mem_free, &d_mem_total);

        size_t d_x_s = 0; if(!x_on_device) d_x_s = 1;
        size_t d_w_s = 0; if(!w_on_device) d_w_s = 1;
        size_t d_b_s = 0; if(!b_on_device) d_b_s = 1;
        size_t d_y_s = 0; if(!y_on_device) d_y_s = 1;

        int n_step_opt = 1, y_c_step_opt = 1, x_c_step_opt = 1, h_step_opt = 1;

        tmp = 1;
        for(n_step_opt = 1; n_step_opt <= n; n_step_opt++){
                time = model_conv_addbias_activ(n, x_c, x_h, x_w, w_h, w_w, y_c, y_h, y_w, d_x_s, d_w_s, d_b_s, d_y_s, 
                                   stride_h, dilation_h, data_size, d_mem_free, n_step_opt, y_c_step_opt, x_c_step_opt, h_step_opt);
                if(time < time_opt){
                        time_opt = time;
                        tmp = n_step_opt;
                }
        }
        if(time_opt > DBL_MAX / 2.0) tmp = n;
        n_step_opt = tmp;

        tmp = 1;
        for(h_step_opt = 1; h_step_opt <= y_h; h_step_opt++){
                time = model_conv_addbias_activ(n, x_c, x_h, x_w, w_h, w_w, y_c, y_h, y_w, d_x_s, d_w_s, d_b_s, d_y_s, 
                                   stride_h, dilation_h, data_size, d_mem_free, n_step_opt, y_c_step_opt, x_c_step_opt, h_step_opt);
                if(time < time_opt){
                        time_opt = time;
                        tmp = h_step_opt;
                }
        }
        if(time_opt > DBL_MAX / 2.0) tmp = y_h;
        h_step_opt = tmp;

        tmp = 1;
        for(x_c_step_opt = 1; x_c_step_opt <= x_c; x_c_step_opt++){
                time = model_conv_addbias_activ(n, x_c, x_h, x_w, w_h, w_w, y_c, y_h, y_w, d_x_s, d_w_s, d_b_s, d_y_s, 
                                   stride_h, dilation_h, data_size, d_mem_free, n_step_opt, y_c_step_opt, x_c_step_opt, h_step_opt);
                if(time < time_opt){
                        time_opt = time;
                        tmp = x_c_step_opt;
                }
        }
        if(time_opt > DBL_MAX / 2.0) tmp = x_c;
        x_c_step_opt = tmp;

        tmp = 1;
        for(y_c_step_opt = 1; y_c_step_opt <= y_c; y_c_step_opt++){
                time = model_conv_addbias_activ(n, x_c, x_h, x_w, w_h, w_w, y_c, y_h, y_w, d_x_s, d_w_s, d_b_s, d_y_s, 
                                   stride_h, dilation_h, data_size, d_mem_free, n_step_opt, y_c_step_opt, x_c_step_opt, h_step_opt);
                if(time < time_opt){
                        time_opt = time;
                        tmp = y_c_step_opt;
                }
        }
        if(time_opt > DBL_MAX / 2.0) tmp = y_c;
        y_c_step_opt = tmp;

    

        *n_step = n_step_opt;
        *y_c_step = y_c_step_opt;
        *x_c_step = x_c_step_opt;
        *h_step = h_step_opt;

        return CUDNN_STATUS_SUCCESS;
}