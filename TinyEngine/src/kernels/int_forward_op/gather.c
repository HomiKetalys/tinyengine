#include "arm_nnfunctions.h"
#include "tinyengine_function.h"



tinyengine_status mgather(int size, const int8_t* input1_data,const int32_t* input2_data,int ch1,int ch2,int8_t* output_data) {
  int k=0,l=0;
  for (int i = 0; i < size; i++) {
      uint8_t *input1_data_=&input1_data[i*ch1];
      for(int j=0;j<ch2;j++)
      {
          output_data[l++]=input1_data_[input2_data[j]];
      }
  }
}

//tinyengine_status mgather(int size, const int8_t* input1_data,const int32_t* input2_data,int ch1,int ch2,int8_t* output_data) {
//    uint8_t *input1_data_=input1_data;
//    uint8_t *output_data_=output_data;
//    for (int i = 0; i++ < size;) {
//      for(int j=0;j<ch2;j++)
//      {
//          output_data_[j]=input1_data_[input2_data[j]];
//      }
//      input1_data_+=ch1;
//      output_data_+=ch2;
//  }
//}
