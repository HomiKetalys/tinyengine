#include "arm_nnfunctions.h"
#include "tinyengine_function.h"



tinyengine_status mgather(int size, const int8_t* input1_data,const int32_t* input2_data,int ch1,int ch2,int8_t* output_data) {
  int j=0;
  for (int i = 0; i < size; ++i) {
      int k=i*ch1;
      for(int c=0;c<ch2;++c)
      {
          output_data[j++]=input1_data[k+input2_data[c]];
      }
  }
}
