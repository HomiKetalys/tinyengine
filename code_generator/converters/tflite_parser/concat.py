import math

import numpy as np

from common_utils.tinyengine.code_generator.operators import concat
from common_utils.tinyengine.code_generator.tflite import Model

from .utils import get_input_tensors, get_nhwc_from_shape, get_output_tensors, getOpCodeStr, getTensorTypeStr


def parse_concat(op, model: Model.Model):
    # operator
    op_code_str = getOpCodeStr(op, model)

    # get input, weight, and output tensors
    input_tensors = get_input_tensors(op, model)
    input_tensor_count = len(input_tensors)
    # assert input_tensor_count == 2, "input should be 2 tensors"
    if input_tensor_count==2:
        input_tensor = input_tensors[0]
        input2_tensor = input_tensors[1]

        output_tensors = get_output_tensors(op, model)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # shapes
        _, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
        _, input2_h, input2_w, input2_c = get_nhwc_from_shape(input2_tensor.tensor.ShapeAsNumpy())
        _, output_h, output_w, output_c = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())
        assert input_h == input2_h == output_h, "tensor shpae not consistent"
        assert input_w == input2_w == output_w, "tensor shpae not consistent"
        assert input_c + input2_c == output_c, "tensor shpae not consistent"

        # tensor types
        input_type = getTensorTypeStr(input_tensor.tensor.Type())
        input_type2 = getTensorTypeStr(input2_tensor.tensor.Type())
        output_type = getTensorTypeStr(output_tensor.tensor.Type())
        assert input_type == input_type2 == output_type, "tensor type not consistent"
        assert input_type=="int8","only support int8"

        # assign params
        params = {
            # operator
            "op": op_code_str,
            # tensor
            "input_idx": input_tensor.tensor_idx,
            "input2_idx": input2_tensor.tensor_idx,
            "output_idx": output_tensor.tensor_idx,
            "input_h": input_h,
            "input_w": input_w,
            "input_c": input_c,
            "input2_h": input_h,
            "input2_w": input_w,
            "input2_c": input2_c,
            "input_dim": 3,
            "input2_dim": 3,
            "output_dim": 3,
            "output_h": output_h,
            "output_w": output_w,
            "output_c": output_c,
            "dtypte": input_type,
        }
        op = concat.Concat2(params)
    elif input_tensor_count==3:
        input_tensor = input_tensors[0]
        input2_tensor = input_tensors[1]
        input3_tensor = input_tensors[2]

        output_tensors = get_output_tensors(op, model)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # shapes
        _, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
        _, input2_h, input2_w, input2_c = get_nhwc_from_shape(input2_tensor.tensor.ShapeAsNumpy())
        _, input3_h, input3_w, input3_c = get_nhwc_from_shape(input3_tensor.tensor.ShapeAsNumpy())
        _, output_h, output_w, output_c = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())
        assert input_h == input2_h == input3_h ==output_h, "tensor shpae not consistent"
        assert input_w == input2_w == input3_h ==output_w, "tensor shpae not consistent"
        assert input_c + input2_c+input3_c == output_c, "tensor shpae not consistent"

        # tensor types
        input_type = getTensorTypeStr(input_tensor.tensor.Type())
        input_type2 = getTensorTypeStr(input2_tensor.tensor.Type())
        input_type3 = getTensorTypeStr(input3_tensor.tensor.Type())
        output_type = getTensorTypeStr(output_tensor.tensor.Type())
        assert input_type == input_type2 ==input_type3 == output_type, "tensor type not consistent"
        assert input_type == "int8", "only support int8"

        # assign params
        params = {
            # operator
            "op": op_code_str,
            # tensor
            "input_idx": input_tensor.tensor_idx,
            "input2_idx": input2_tensor.tensor_idx,
            "input3_idx": input3_tensor.tensor_idx,
            "output_idx": output_tensor.tensor_idx,
            "input_h": input_h,
            "input_w": input_w,
            "input_c": input_c,
            "input2_h": input_h,
            "input2_w": input_w,
            "input2_c": input2_c,
            "input3_h": input_h,
            "input3_w": input_w,
            "input3_c": input3_c,
            "input_dim": 3,
            "input2_dim": 3,
            "input3_dim": 3,
            "output_dim": 3,
            "output_h": output_h,
            "output_w": output_w,
            "output_c": output_c,
            "dtypte": input_type,
        }
        op = concat.Concat3(params)
    elif input_tensor_count==4:
        input_tensor = input_tensors[0]
        input2_tensor = input_tensors[1]
        input3_tensor = input_tensors[2]
        input4_tensor = input_tensors[3]

        output_tensors = get_output_tensors(op, model)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # shapes
        _, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
        _, input2_h, input2_w, input2_c = get_nhwc_from_shape(input2_tensor.tensor.ShapeAsNumpy())
        _, input3_h, input3_w, input3_c = get_nhwc_from_shape(input3_tensor.tensor.ShapeAsNumpy())
        _, input4_h, input4_w, input4_c = get_nhwc_from_shape(input4_tensor.tensor.ShapeAsNumpy())
        _, output_h, output_w, output_c = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())
        assert input_h == input2_h == input3_h ==input4_h ==output_h, "tensor shpae not consistent"
        assert input_w == input2_w == input3_w ==input4_w ==output_w, "tensor shpae not consistent"
        assert input_c + input2_c+input3_c +input4_c== output_c, "tensor shpae not consistent"

        # tensor types
        input_type = getTensorTypeStr(input_tensor.tensor.Type())
        input_type2 = getTensorTypeStr(input2_tensor.tensor.Type())
        input_type3 = getTensorTypeStr(input3_tensor.tensor.Type())
        input_type4 = getTensorTypeStr(input4_tensor.tensor.Type())
        output_type = getTensorTypeStr(output_tensor.tensor.Type())
        assert input_type == input_type2 ==input_type3 == input_type4 ==output_type, "tensor type not consistent"
        assert input_type == "int8", "only support int8"

        # assign params
        params = {
            # operator
            "op": op_code_str,
            # tensor
            "input_idx": input_tensor.tensor_idx,
            "input2_idx": input2_tensor.tensor_idx,
            "input3_idx": input3_tensor.tensor_idx,
            "input4_idx": input4_tensor.tensor_idx,
            "output_idx": output_tensor.tensor_idx,
            "input_h": input_h,
            "input_w": input_w,
            "input_c": input_c,
            "input2_h": input_h,
            "input2_w": input_w,
            "input2_c": input2_c,
            "input3_h": input_h,
            "input3_w": input_w,
            "input3_c": input3_c,
            "input4_h": input_h,
            "input4_w": input_w,
            "input4_c": input4_c,
            "input_dim": 3,
            "input2_dim": 3,
            "input3_dim": 3,
            "input4_dim": 3,
            "output_dim": 3,
            "output_h": output_h,
            "output_w": output_w,
            "output_c": output_c,
            "dtypte": input_type,
        }
        op = concat.Concat4(params)

    return op
