/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   add_fpreq.c
 *
 * Reference papers:
 *  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
 *  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
 *  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
 * Contact authors:
 *  - Wei-Ming Chen, wmchen@mit.edu
 *  - Wei-Chen Wang, wweichen@mit.edu
 *  - Ji Lin, jilin@mit.edu
 *  - Ligeng Zhu, ligeng@mit.edu
 *  - Song Han, songhan@mit.edu
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include <math.h>
#include "arm_math.h"
#include "tinyengine_function.h"

//tinyengine_status add_fpreq(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
//			const int8_t* input2_data, const float input2_scale, const float input2_zero, const float output_scale,
//			const float zero_y, int8_t* output_data) {
//  for (int i = 0; i < size; ++i) {
//	  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
//	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
//      int clamped_output = (int)roundf((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
//      clamped_output = TN_MAX(clamped_output, -128);
//      clamped_output = TN_MIN(clamped_output, 127);
//      output_data[i] = (int8_t)(clamped_output);
//  }
//}

//tinyengine_status add_fpreq(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
//			const int8_t* input2_data, const float input2_scale, const float input2_zero, const float output_scale,
//			const float zero_y, int8_t* output_data) {
//  float sc1=input1_scale/output_scale;
//  float sc2=input2_scale/output_scale;
//  float offset=zero_y-(input1_zero*sc1+input2_zero*sc2);
//  for (int i = 0; i < size;) {
//      int clamped_output = (int)roundf((*input1_data++)*sc1+(*input2_data++)*sc2 + offset); // to align with tvm implementation
//      clamped_output = TN_MAX(clamped_output, -128);
//      clamped_output = TN_MIN(clamped_output, 127);
//      output_data[i++] = (int8_t)(clamped_output);
//  }
//}

tinyengine_status add_fpreq(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
			const int8_t* input2_data, const float input2_scale, const float input2_zero, const float output_scale,
			const float zero_y, int8_t* output_data) {
    float scale1=input1_scale/output_scale;
    float scale2=input2_scale/output_scale;
    float offset=(float)zero_y-((float)input1_zero*scale1+(float)input2_zero*scale2);
    int8_t* final_output_data=output_data+4*(size/4);
    while(output_data<final_output_data) {
        int8_t in01,in02,in11,in12,in21,in22,in31,in32;
        in01=input1_data[0];
        in11=input1_data[1];
        in21=input1_data[2];
        in31=input1_data[3];
        in02=input2_data[0];
        in12=input2_data[1];
        in22=input2_data[2];
        in32=input2_data[3];
        int32_t out0,out1,out2,out3;
        out0=(int32_t)roundf(((float)in01*scale1+(float)in02*scale2 + offset));
        out1=(int32_t)roundf(((float)in11*scale1+(float)in12*scale2 + offset));
        out2=(int32_t)roundf(((float)in21*scale1+(float)in22*scale2 + offset));
        out3=(int32_t)roundf(((float)in31*scale1+(float)in32*scale2 + offset));
        __asm volatile (
        "SSAT %0, #8, %1"
        : "=r" (out0)
        : "r" (out0)
        );
        __asm volatile (
        "SSAT %0, #8, %1"
        : "=r" (out1)
        : "r" (out1)
        );
        __asm volatile (
        "SSAT %0, #8, %1"
        : "=r" (out2)
        : "r" (out2)
        );
        __asm volatile (
        "SSAT %0, #8, %1"
        : "=r" (out3)
        : "r" (out3)
        );
        output_data[0] = (int8_t)out0;
        output_data[1] = (int8_t)out1;
        output_data[2] = (int8_t)out2;
        output_data[3] = (int8_t)out3;
        input1_data+=4;
        input2_data+=4;
        output_data+=4;
    }
    output_data=final_output_data;
    final_output_data=final_output_data+size%4;
    while(output_data<final_output_data)
    {
        int8_t in01,in02;
        in01=*(input1_data++);
        in02=*(input2_data++);
        int32_t out0;
        out0=(int32_t)roundf(((float)in01*scale1+(float)in02*scale2 + offset));
        __asm volatile (
        "SSAT %0, #8, %1"
        : "=r" (out0)
        : "r" (out0)
        );
        *output_data++ = (int8_t)out0;
    }
}


const int activation_min = -128;
const int activation_max = 127;
tinyengine_status add_fpreq_mask(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
			const int8_t* input2_data, const float input2_scale, const float input2_zero, const float output_scale,
			const float zero_y, int8_t* output_data, int8_t* output_mask) {
  for (int i = 0; i < size; ++i) {
	  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      int8_t mask_value = 1;
	  if (clamped_output < activation_min){
		  clamped_output = activation_min;
		  mask_value = 0;
	  }
	  if (clamped_output > activation_max){
		  clamped_output = activation_max;
		  mask_value = 0;
	  }
      output_data[i] = (int8_t)(clamped_output);
      output_mask[i] = mask_value;
  }
}


tinyengine_status add_fpreq_bitmask(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
			const int8_t* input2_data, const float input2_scale, const float input2_zero, const float output_scale,
			const float zero_y, int8_t* output_data, int8_t* output_mask) {
  int mask_idx = 0;
  for (int i = 0; i < size; ++i) {
	  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      int8_t mask_value = 1;
	  if (clamped_output < activation_min){
		  clamped_output = activation_min;
		  mask_value = 0;
	  }
	  if (clamped_output > activation_max){
		  clamped_output = activation_max;
		  mask_value = 0;
	  }
      output_data[i] = (int8_t)(clamped_output);
	  if (mask_value == 1)
		  BIT_SET(*output_mask, mask_idx);
	  else
		  BIT_CLEAR(*output_mask, mask_idx);
	  mask_idx++;
	  if (mask_idx == 8){
		  mask_idx = 0;
		  output_mask++;
	  }
  }
}
