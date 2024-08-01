#include <math.h>
#include "arm_math.h"
#include "tinyengine_function.h"

tinyengine_status msigmoid(int size, const int8_t* input_data, const float input_scale, const int32_t input_zero,
            const float output_scale,const int32_t zero_y, int8_t* output_data) {
  for (int i = 0; i < size; ++i) {
      int32_t val=roundf(1.f/(1.f+expf(-input_scale*(*input_data++-input_zero)))/output_scale+zero_y);
      val = TN_MAX(val, -128);
      val = TN_MIN(val, 127);
      output_data[i] = val;

  }
}

tinyengine_status mtanh(int size, const int8_t* input_data, const float input_scale, const int32_t input_zero,
            const float output_scale,const int32_t zero_y, int8_t* output_data) {
  for (int i = 0; i < size; ++i) {
      int32_t val=roundf(tanhf(input_scale*(*input_data++-input_zero))/output_scale+zero_y);
      val = TN_MAX(val, -128);
      val = TN_MIN(val, 127);
      output_data[i] = val;
  }
}

tinyengine_status mquantize(int size, const int8_t* input_data, const float input_scale, const int32_t input_zero,
            const float output_scale,const int32_t zero_y, int8_t* output_data) {
  float scale=input_scale/output_scale;
  for (int i = 0; i < size; ++i) {
      int32_t val=roundf((*input_data++-input_zero)*scale+zero_y);
      val = TN_MAX(val, -128);
      val = TN_MIN(val, 127);
      output_data[i] = val;
  }
}

tinyengine_status mdequantize(int size, const int8_t* input_data, const float input_scale, const int32_t input_zero,
            float* output_data) {
  for (int i = 0; i < size; ++i) {
      output_data[i] = (*input_data++-input_zero)*input_scale;
  }
}