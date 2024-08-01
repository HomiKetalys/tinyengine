import math

import numpy as np

from common_utils.tinyengine.code_generator.operators import gather
from common_utils.tinyengine.code_generator.tflite import Model
from common_utils.tinyengine.code_generator.tflite.GatherOptions import GatherOptions

from .utils import get_input_tensors, get_nhwc_from_shape, get_output_tensors, getOpCodeStr, getTensorTypeStr
from ...tflite.BuiltinOptions import BuiltinOptions


def parse_gather(op, model: Model.Model):
    # operator
    op_code_str = getOpCodeStr(op, model)

    # get input, weight, and output tensors
    input_tensors = get_input_tensors(op, model)
    input_tensor_count = len(input_tensors)
    assert input_tensor_count == 2, "input should be 2 tensors"

    input_tensor = input_tensors[0]
    input2_tensor = input_tensors[1]

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    assert op.BuiltinOptionsType() == BuiltinOptions.GatherOptions
    op_options = op.BuiltinOptions()
    gather_options = GatherOptions()
    gather_options.Init(op_options.Bytes, op_options.Pos)
    assert gather_options.Axis()==-1 or gather_options.Axis()==2,"only support channel dim"


    # shapes
    _, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
    _, input2_h, input2_w, input2_c = get_nhwc_from_shape(input2_tensor.tensor.ShapeAsNumpy())
    _, output_h, output_w, output_c = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())
    assert input_h  == output_h, "tensor shpae not consistent"
    assert input_w  == output_w, "tensor shpae not consistent"
    assert input2_c ==output_c,"tensor shpae not consistent"

    # tensor types
    input_type = getTensorTypeStr(input_tensor.tensor.Type())
    output_type = getTensorTypeStr(output_tensor.tensor.Type())
    assert input_type  == output_type, "tensor type not consistent"
    assert input_type=="int8","only support int8"

    poses=np.frombuffer(input2_tensor.buffer.DataAsNumpy().tobytes(),dtype="int64")

    # assign params
    params = {
        # operator
        "op": op_code_str,
        # tensor
        "input_idx": input_tensor.tensor_idx,
        # "input2_idx": input_tensor.tensor_idx,
        "output_idx": output_tensor.tensor_idx,
        "input_h": input_h,
        "input_w": input_w,
        "input_c": input_c,
        # "input2_c": input2_c,
        "input_dim": 3,
        "output_dim": 3,
        "output_h": output_h,
        "output_w": output_w,
        "output_c": output_c,
        "dtypte": input_type,
        "poses":poses
    }
    op = gather.Gather(params)

    return op
