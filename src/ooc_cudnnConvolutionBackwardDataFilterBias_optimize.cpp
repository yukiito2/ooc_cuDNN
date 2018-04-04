

#include "ooc_cudnn.h"

#include <sys/time.h>
#include <float.h>


extern double cudnnConvolutionBackwardData_weight;
extern double cudnnConvolutionBackwardData_bias;
extern double cudnnConvolutionBackwardFilter_weight;
extern double cudnnConvolutionBackwardFilter_bias;
extern double cudnnConvolutionBackwardBias_weight;
extern double cudnnConvolutionBackwardBias_bias;
extern double cudaMemcpy_weight;
extern double cudaMemcpy_bias;


double model_convbackdatafilterbias(
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
                                const size_t                        d_dy_s,
                                const size_t                        d_dx_s,
                                const size_t                        d_dw_s,
                                const size_t                        d_db_s,
                                const size_t                        stride_h,
                                const size_t                        dilation_h,
                                const size_t                        data_size,
                                const size_t                        d_mem_free,
                                const int                           n_step,
                                const int                           x_c_step,
                                const int                           y_c_step,
                                const int                           h_step ){

        double t = DBL_MAX;
    
        int dilation_w_h = (w_h - 1) * dilation_h + 1;
    
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

        size_t d_mem_x = d_n * d_x_c * d_x_h * x_w * data_size * d_x_s;
        size_t d_mem_w = d_y_c * d_x_c * w_h * w_w * data_size * d_w_s;
        size_t d_mem_dy = d_n * d_y_c * d_y_h * y_w * data_size * d_dy_s;
        size_t d_mem_dx = d_n * d_x_c * d_x_h * x_w * data_size * d_dx_s;
        size_t d_mem_dx2 = d_n2 * d_x_c2 * d_x_h2 * x_w * data_size * d_dx_s;
        size_t d_mem_dw = d_y_c * d_x_c * w_h * w_w * data_size * d_dw_s;
        size_t d_mem_dw2 = d_y_c2 * d_x_c2 * w_h * w_w * data_size * d_dw_s;
        size_t d_mem_db = 1 * d_y_c * 1 * 1 * data_size * d_db_s;
        size_t d_mem_db2 = 1 * d_y_c2 * 1 * 1 * data_size * d_db_s;


        if(d_y_c % 32 != 0) d_y_c = 32 * (d_y_c / 32 + 1);
        size_t comp_conv1 = d_n * d_x_c * (d_x_h / stride_h) * y_w * d_y_c * w_h * w_w * data_size;
        size_t comp_conv2 = d_n * d_y_c * d_y_h * y_w * data_size;
        if(d_y_c2 % 32 != 0) d_y_c2 = 32 * (d_y_c2 / 32 + 1);
        size_t comp_conv1_2 = d_n2 * d_x_c2 * (d_x_h2 / stride_h) * y_w * d_y_c2 * w_h * w_w * data_size;
        size_t comp_conv2_2 = d_n2 * d_y_c2 * d_y_h2 * y_w * data_size;


        if(d_mem_x + d_mem_w + d_mem_dy + d_mem_dx + d_mem_dw + d_mem_db < d_mem_free / 3){
                double t_pre1 = cudaMemcpy_weight * d_mem_x + cudaMemcpy_bias * d_x_s;
                double t_pre2 = cudaMemcpy_weight * (d_mem_w + d_mem_dy + d_mem_dw + d_mem_db) + cudaMemcpy_bias * (d_w_s + d_dy_s + d_dw_s + d_db_s);
                double t_conv = (cudnnConvolutionBackwardData_weight + cudnnConvolutionBackwardFilter_weight) * comp_conv1 + cudnnConvolutionBackwardBias_weight * comp_conv2
                              + cudnnConvolutionBackwardData_bias + cudnnConvolutionBackwardFilter_bias + cudnnConvolutionBackwardBias_bias;
                double t_conv2 = (cudnnConvolutionBackwardData_weight + cudnnConvolutionBackwardFilter_weight) * comp_conv1_2 + cudnnConvolutionBackwardBias_weight * comp_conv2_2
                               + cudnnConvolutionBackwardData_bias + cudnnConvolutionBackwardFilter_bias + cudnnConvolutionBackwardBias_bias;
                double t_off1 = cudaMemcpy_weight * (d_mem_dw + d_mem_db) + cudaMemcpy_bias * (d_dw_s + d_db_s);
                double t_off2 = cudaMemcpy_weight * d_mem_dx + cudaMemcpy_bias * d_dx_s;
                double t_off1_2 = cudaMemcpy_weight * (d_mem_dw2 + d_mem_db2) + cudaMemcpy_bias * (d_dw_s + d_db_s);
                double t_off2_2 = cudaMemcpy_weight * d_mem_dx2 + cudaMemcpy_bias * d_dx_s;


                double steps = (n_step * x_c_step * h_step);
                double t_tmp;
                
                t = steps * (t_pre1 + y_c_step * t_pre2) + t_conv2 + t_off1_2 + t_off2_2;
                
                t_tmp = t_pre1 + t_pre2 + steps * y_c_step * t_conv + t_off1_2 + t_off2_2;
                if(t < t_tmp) t = t_tmp;

                t_tmp = t_pre1 + y_c_step * (t_pre2 + t_conv) + steps * (y_c_step * t_off1 + t_off2);
                if(t < t_tmp) t = t_tmp;
        }

        return t;
}


cudnnStatus_t CUDNNWINAPI ooc_cudnnConvolutionBackwardDataFilterBias_optimize(
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
                                const bool                          dy_on_device,
                                const bool                          dx_on_device,
                                const bool                          dw_on_device,
                                const bool                          db_on_device,
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

        size_t d_x_s = 0; if(!x_on_device) d_x_s = 1;
        size_t d_w_s = 0; if(!w_on_device) d_w_s = 1;
        size_t d_dy_s = 0; if(!dy_on_device) d_dy_s = 1;
        size_t d_dx_s = 0; if(!dx_on_device) d_dx_s = 1;
        size_t d_dw_s = 0; if(!dw_on_device) d_dw_s = 1;
        size_t d_db_s = 0; if(!db_on_device) d_db_s = 1;


        int n_step_opt = 1, y_c_step_opt = 1, x_c_step_opt = 1, h_step_opt = 1;

        tmp = 1;
        for(n_step_opt = 1; n_step_opt <= n; n_step_opt++){
                time = model_convbackdatafilterbias(n, x_c, x_h, x_w, w_h, w_w, y_c, y_h, y_w, 
                                   d_x_s, d_w_s, d_dy_s, d_dx_s, d_dw_s, d_db_s, 
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
                time = model_convbackdatafilterbias(n, x_c, x_h, x_w, w_h, w_w, y_c, y_h, y_w, 
                                   d_x_s, d_w_s, d_dy_s, d_dx_s, d_dw_s, d_db_s, 
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
                time = model_convbackdatafilterbias(n, x_c, x_h, x_w, w_h, w_w, y_c, y_h, y_w, 
                                   d_x_s, d_w_s, d_dy_s, d_dx_s, d_dw_s, d_db_s, 
                                   stride_h, dilation_h, data_size, d_mem_free, n_step_opt, x_c_step_opt, y_c_step_opt, h_step_opt);
                if(time < time_opt){
                        time_opt = time;
                        tmp = x_c_step_opt;
                }
        }
        if(time_opt > DBL_MAX / 2.0) tmp = x_c;
        x_c_step_opt = tmp;

        *n_step = n_step_opt;
        *x_c_step = x_c_step_opt;
        *y_c_step = y_c_step_opt;
        *h_step = h_step_opt;

        return CUDNN_STATUS_SUCCESS;
}