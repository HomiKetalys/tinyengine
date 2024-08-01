import math

import numpy as np

from common_utils.tinyengine.code_generator.operators import dequantize
from common_utils.tinyengine.code_generator.tflite import Model
from common_utils.tinyengine.code_generator.tflite.GatherOptions import GatherOptions

from .utils import get_input_tensors, get_nhwc_from_shape, get_output_tensors, getOpCodeStr, getTensorTypeStr
from ...tflite.BuiltinOptions import BuiltinOptions


def parse_dequantize(op, model: Model.Model,last_id=None):
    # operator
    op_code_str = getOpCodeStr(op, model)

    # get input, weight, and output tensors
    input_tensors = get_input_tensors(op, model)
    input_tensor_count = len(input_tensors)
    assert input_tensor_count == 1, "input should be 1 tensors"

    input_tensor = input_tensors[0]

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    # shapes
    _, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
    _, output_h, output_w, output_c = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())
    assert input_h  == output_h, "tensor shpae not consistent"
    assert input_w  == output_w, "tensor shpae not consistent"

    # tensor types
    input_type = getTensorTypeStr(input_tensor.tensor.Type())
    output_type = getTensorTypeStr(output_tensor.tensor.Type())
    # assert input_type  == output_type, "tensor type not consistent"
    assert input_type=="int8","only support int8"

    input_zero_point = None
    input_scale = None
    output_zero_point = None
    output_scale = None

    if input_type != "float32":
        input_zero_point = input_tensor.qnn_params["zero_point"]
        input_scale = input_tensor.qnn_params["scale"]
    # if output_type != "float32":
    #     output_zero_point = output_tensor.qnn_params["zero_point"]
    #     output_scale = output_tensor.qnn_params["scale"]

    # assign params
    params = {
        # operator
        "op": op_code_str,
        # tensor
        "input_idx": input_tensor.tensor_idx if last_id is None else last_id,
        "output_idx": output_tensor.tensor_idx,
        "input_h": input_h,
        "input_w": input_w,
        "input_c": input_c,
        "input_dim": 3,
        "output_dim": 3,
        "output_h": output_h,
        "output_w": output_w,
        "output_c": output_c,
        "dtypte": input_type,
        # trainable parameters
        "input_zero_point": input_zero_point,
        "output_zero_point": output_zero_point,
        "input_scale": input_scale,
        "output_scale": output_scale,
    }
    op = dequantize.Dequantize(params)

    return op
