/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   maxpooling.c
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

#include "tinyengine_function.h"

tinyengine_status max_pooling_downsample(const q7_t* input, const uint16_t input_h, const uint16_t input_w,
		const uint16_t input_c,	const uint16_t sample_h, const uint16_t sample_w,
		const uint16_t output_h, const uint16_t output_w, const int32_t out_activation_min,
        const int32_t out_activation_max, q7_t* output)
{
	int h, w, c;
	int sh, sw;
	for(c = 0; c < input_c; c++){
		for(h = 0; h < output_h; h++){
			for(w = 0; w < output_w; w++){
				int max = out_activation_min;

				for(sh = 0; sh < sample_h; sh++){
					int height = sh + h * sample_h;
					for(sw = 0; sw < sample_w; sw++){
						int width = sw + w * sample_w;
						max = TN_MAX(max,input[(width + height * input_w) * input_c + c]);
					}
				}

				int out = max;
				out = TN_MAX(out, out_activation_min);
				out = TN_MIN(out, out_activation_max);
				output[(w + h * output_w) * input_c + c] = out;
			}
		}
	}
}

tinyengine_status max_pooling(const q7_t* input, const uint16_t input_h, const uint16_t input_w,
		const uint16_t input_c,	const uint16_t sample_h, const uint16_t sample_w,const int32_t offset,
		const uint16_t output_h, const uint16_t output_w, const int32_t out_activation_min,
        const int32_t out_activation_max, q7_t* output)
{
	int h, w, c;
	int sh, sw;
	for(c = 0; c < input_c; c++){
		for(h = 0; h < output_h; h++){
			for(w = 0; w < output_w; w++){
				int max = out_activation_min;
                max=(h<sample_h/2||output_h-1-h<sample_h/2||w<sample_w/2||output_w-1-w<sample_w/2)?offset:max;

				for(sh = -sample_h/2; sh <= sample_h/2; sh++){
					int height = TN_MIN(TN_MAX(sh + h ,0),output_h-1);
					for(sw = -sample_w/2; sw <= sample_w/2; sw++){
						int width = TN_MIN(TN_MAX(sw + w ,0),output_w-1);
						max = TN_MAX(max,input[(width + height * input_w) * input_c + c]);
					}
				}

				int out = max;
				out = TN_MAX(out, out_activation_min);
				out = TN_MIN(out, out_activation_max);
				output[(w + h * output_w) * input_c + c] = out;
			}
		}
	}
}


