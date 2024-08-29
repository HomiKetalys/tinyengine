#include <math.h>
#include "arm_math.h"
#include "tinyengine_function.h"



#define MAX_TABLE_NUM 20
uint8_t table_init[MAX_TABLE_NUM]={0};


void table_init_func(const uint8_t* table_buffer,const int32_t table_id,const float input_scale, const int32_t input_zero,const float output_scale,const int32_t zero_y,float (*method)(int8_t,const float,const int32_t,const float,int32_t))
{
    uint32_t id=table_id;
    if(table_init[id])
        return;
    uint8_t *table_buffer_=table_buffer+id*256;
    for(int i=0;i<256;)
    {
        int32_t val=roundf(method(i,input_scale,input_zero,output_scale,zero_y));
        val = TN_MAX(val, -128);
        val = TN_MIN(val, 127);
        table_buffer_[i++]=val;
    }
    table_init[id]=1;
}
#define ELE_NUM 4
void move_buffer(int size,uint8_t *input_data,uint8_t *output_data,uint8_t *buffer)
{
  uint32_t a,b;
  uint32_t *input_data_=input_data;
//  uint8_t inp_buffer[ELE_NUM];
  size/=ELE_NUM;

  for (int i = 0; i++ < size;) {
//      uint8_t *inp_buffer_=&inp_buffer[0];
//      #pragma GCC unroll 4
//      for(int j=0;j<ELE_NUM;)
//          inp_buffer_[j++]=*input_data++;
//
//      #pragma GCC unroll 4
//      for(int j=0;j<ELE_NUM;)
//          output_data[j++] = buffer[*inp_buffer_++];

      a=*input_data_++;
      output_data[0] = buffer[(uint8_t)a];
      output_data[1] = buffer[(uint8_t)(a>>8)];
      output_data[2] = buffer[(uint8_t)(a>>16)];
      output_data[3] = buffer[(uint8_t)(a>>24)];
      output_data+=ELE_NUM;
  }
}

float ele_sigmoid(int8_t inp,const float input_scale, const int32_t input_zero,const float output_scale,const int32_t zero_y)
{
    return 1.f/(1.f+expf(-input_scale*(inp-input_zero)))/output_scale+zero_y;
}

tinyengine_status msigmoid(int size, const int8_t* input_data, const float input_scale, const int32_t input_zero,
            const float output_scale,const int32_t zero_y, const int8_t* table_buffer,const int32_t table_id,int8_t* output_data) {
  table_init_func(table_buffer,table_id,input_scale,input_zero,output_scale,zero_y,ele_sigmoid);
  uint8_t *buffer=&table_buffer[table_id*256];
  uint8_t *input_data_=input_data;
  uint8_t *output_data_=output_data;
  move_buffer(size,input_data_,output_data_,buffer);
}


tinyengine_status msoftmax(int size, int ch,const int8_t* input_data, const float input_scale, const int32_t input_zero,
            const float output_scale,const int32_t zero_y, int8_t* output_data) {
    int i;
    float buf[256];
    for(i=0;i<256;i++)
    {
        int8_t val=i;
        buf[i]=expf(input_scale*((int32_t)val-input_zero));
    }
    for(i=0;i<size;i++)
    {
        int j;
        float sum=0;
        for(j=0;j<ch;j++)
        {
            uint8_t val=input_data[j];
            sum+=buf[val];
        }
        sum*=output_scale;
        for(j=0;j<ch;j++)
        {
            uint8_t val=input_data[j];
            int32_t out=(int32_t)roundf(buf[val]/sum)+zero_y;
            out = TN_MAX(out, -128);
            out = TN_MIN(out, 127);
            output_data[j]=(int8_t) out;
        }
        input_data+=ch;
        output_data+=ch;
    }
}

float ele_tanh(int8_t inp,const float input_scale, const int32_t input_zero,const float output_scale,const int32_t zero_y)
{
    return tanhf(input_scale*(inp-input_zero))/output_scale+zero_y;
}

tinyengine_status mtanh(int size, const int8_t* input_data, const float input_scale, const int32_t input_zero,
            const float output_scale,const int32_t zero_y, const int8_t* table_buffer,const int32_t table_id,int8_t* output_data) {
  table_init_func(table_buffer,table_id,input_scale,input_zero,output_scale,zero_y,ele_tanh);
  uint8_t *buffer=&table_buffer[table_id*256];
  uint8_t *input_data_=input_data;
  uint8_t *output_data_=output_data;
//  #pragma GCC unroll 4
  for (int i = 0; i < size; ) {
      output_data_[i++] = buffer[*input_data_++];
  }
}

float ele_quantize(int8_t inp,const float input_scale, const int32_t input_zero,const float output_scale,const int32_t zero_y)
{
    return (inp-input_zero)*input_scale/output_scale+zero_y;
}


tinyengine_status mquantize(int size, const int8_t* input_data, const float input_scale, const int32_t input_zero,
            const float output_scale,const int32_t zero_y,const int8_t* table_buffer,const int32_t table_id, int8_t* output_data) {
  table_init_func(table_buffer,table_id,input_scale,input_zero,output_scale,zero_y,ele_quantize);
  uint8_t *buffer=&table_buffer[table_id*256];
  uint8_t *input_data_=input_data;
  uint8_t *output_data_=output_data;
  move_buffer(size,input_data_,output_data_,buffer);
}

tinyengine_status mdequantize(int size, const int8_t* input_data, const float input_scale, const int32_t input_zero,
            float* output_data) {
  for (int i = 0; i < size;) {
      output_data[i++] = (*input_data++-input_zero)*input_scale;
  }
}