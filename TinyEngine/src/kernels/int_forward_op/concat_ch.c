/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   concat_ch.c
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

#include "arm_nnfunctions.h"
#include "tinyengine_function.h"

tinyengine_status concat_ch(const q7_t *input1, const uint16_t input_x,
	const uint16_t input_y, const uint16_t input1_ch, const q7_t* input2, const uint16_t input2_ch, q7_t *output) {

	int elements = input_y * input_x;

	while(elements--){
		//place the first input
		memcpy(output, input1, input1_ch);
		input1 += input1_ch; output += input1_ch;

		//place the second input
		memcpy(output, input2, input2_ch);
		input2 += input2_ch; output += input2_ch;
	}

	return STATE_SUCCESS;
}

tinyengine_status mconcat2(int size, const int8_t* input1_data,const int8_t* input2_data,int ch1,int ch2,int8_t* output_data) {
  for (int i = 0; i < size; ++i) {
      memcpy(output_data, input1_data, ch1);
      output_data+=ch1;
      input1_data+=ch1;
      memcpy(output_data, input2_data, ch2);
      output_data+=ch2;
      input2_data+=ch2;
  }
}

tinyengine_status mconcat3(int size, const int8_t* input1_data,const int8_t* input2_data,const int8_t* input3_data,int ch1,int ch2,int ch3,int8_t* output_data) {
  for (int i = 0; i < size; ++i) {
      memcpy(output_data, input1_data, ch1);
      output_data+=ch1;
      input1_data+=ch1;
      memcpy(output_data, input2_data, ch2);
      output_data+=ch2;
      input2_data+=ch2;
      memcpy(output_data, input3_data, ch3);
      output_data+=ch3;
      input3_data+=ch3;
  }
}

tinyengine_status mconcat4(int size, const int8_t* input1_data,const int8_t* input2_data,const int8_t* input3_data,const int8_t* input4_data,int ch1,int ch2,int ch3,int ch4,int8_t* output_data) {
  for (int i = 0; i < size; ++i) {
      memcpy(output_data, input1_data, ch1);
      output_data+=ch1;
      input1_data+=ch1;
      memcpy(output_data, input2_data, ch2);
      output_data+=ch2;
      input2_data+=ch2;
      memcpy(output_data, input3_data, ch3);
      output_data+=ch3;
      input3_data+=ch3;
      memcpy(output_data, input4_data, ch4);
      output_data+=ch4;
      input4_data+=ch4;
  }
}
