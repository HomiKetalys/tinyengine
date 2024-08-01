import warnings

from ..constant import USE_BIT_MASK
from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

__all__ = ["Concat2","Concat3","Concat4"]

default_params2 = {
    # op related
    "op": "Concat",
    "input_idx": None,
    "input2_idx": None,
    "output_idx": None,
    # tensor related
    "input_dim": None,
    "input_h": None,
    "input_w": None,
    "input_c": None,
    "input2_dim": None,
    "input2_h": None,
    "input2_w": None,
    "input2_c": None,
    "output_dim": None,
    "output_h": None,
    "output_w": None,
    "output_c": None,
    "input_dtype": "int8",
    "input2_dtype": "int8",
    "output_dtype": "int8",
}


class Concat2(basicOperator):
    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params2)
        overwrite_dicts(self.params, params)
        super().__init__()
        # handle input/output tensors in HWC format
        self._add_input(
            self.params["input_idx"],
            self.params["input_dtype"],
            self.params["input_c"],
            self.params["input_w"],
            self.params["input_h"],
        )
        self._add_input(
            self.params["input2_idx"],
            self.params["input2_dtype"],
            self.params["input2_c"],
            self.params["input2_w"],
            self.params["input2_h"],
        )
        self._add_output(
            self.params["output_idx"],
            self.params["output_dtype"],
            self.params["output_c"],
            self.params["output_w"],
            self.params["output_h"],
        )

        if None in default_params2:
            warnings.warn(f"parameters are not all set for op {self.params['op']}")

    def get_macs(self) -> int:
        return 0

    def generate_inference_str(self):
        string = ""
        params = self.params
        string += (
            f"mconcat2({str(int(params['input_h']*params['input_w']))}, "
            + f"{self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
            + f"{self._getBufferstr(params['input2_buf_add'], params['input2_buf_add_offset'])},"
            + f"{params['input_c']},{params['input2_c']},"
            + f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])});\n"
        )
        return string


default_params3 = {
    # op related
    "op": "Concat",
    "input_idx": None,
    "input2_idx": None,
"input3_idx": None,
    "output_idx": None,
    # tensor related
    "input_dim": None,
    "input_h": None,
    "input_w": None,
    "input_c": None,
    "input2_dim": None,
    "input2_h": None,
    "input2_w": None,
    "input2_c": None,
    "input3_dim": None,
    "input3_h": None,
    "input3_w": None,
    "input3_c": None,
    "output_dim": None,
    "output_h": None,
    "output_w": None,
    "output_c": None,
    "input_dtype": "int8",
    "input2_dtype": "int8",
"input3_dtype": "int8",
    "output_dtype": "int8",
}

class Concat3(basicOperator):
    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params3)
        overwrite_dicts(self.params, params)
        super().__init__()
        # handle input/output tensors in HWC format
        self._add_input(
            self.params["input_idx"],
            self.params["input_dtype"],
            self.params["input_c"],
            self.params["input_w"],
            self.params["input_h"],
        )
        self._add_input(
            self.params["input2_idx"],
            self.params["input2_dtype"],
            self.params["input2_c"],
            self.params["input2_w"],
            self.params["input2_h"],
        )
        self._add_input(
            self.params["input3_idx"],
            self.params["input3_dtype"],
            self.params["input3_c"],
            self.params["input3_w"],
            self.params["input3_h"],
        )
        self._add_output(
            self.params["output_idx"],
            self.params["output_dtype"],
            self.params["output_c"],
            self.params["output_w"],
            self.params["output_h"],
        )

        if None in default_params3:
            warnings.warn(f"parameters are not all set for op {self.params['op']}")

    def get_macs(self) -> int:
        return 0

    def generate_inference_str(self):
        string = ""
        params = self.params
        string += (
            f"mconcat3({str(int(params['input_h']*params['input_w']))}, "
            + f"{self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
            + f"{self._getBufferstr(params['input2_buf_add'], params['input2_buf_add_offset'])},"
            + f"{self._getBufferstr(params['input3_buf_add'], params['input3_buf_add_offset'])},"
            + f"{params['input_c']},{params['input2_c']},{params['input3_c']},"
            + f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])});\n"
        )
        return string

default_params4 = {
    # op related
    "op": "Concat",
    "input_idx": None,
    "input2_idx": None,
"input3_idx": None,
"input4_idx": None,
    "output_idx": None,
    # tensor related
    "input_dim": None,
    "input_h": None,
    "input_w": None,
    "input_c": None,
    "input2_dim": None,
    "input2_h": None,
    "input2_w": None,
    "input2_c": None,
    "input3_dim": None,
    "input3_h": None,
    "input3_w": None,
    "input3_c": None,
    "input4_dim": None,
    "input4_h": None,
    "input4_w": None,
    "input4_c": None,
    "output_dim": None,
    "output_h": None,
    "output_w": None,
    "output_c": None,
    "input_dtype": "int8",
    "input2_dtype": "int8",
"input3_dtype": "int8",
"input4_dtype": "int8",
    "output_dtype": "int8",
}

class Concat4(basicOperator):
    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params4)
        overwrite_dicts(self.params, params)
        super().__init__()
        # handle input/output tensors in HWC format
        self._add_input(
            self.params["input_idx"],
            self.params["input_dtype"],
            self.params["input_c"],
            self.params["input_w"],
            self.params["input_h"],
        )
        self._add_input(
            self.params["input2_idx"],
            self.params["input2_dtype"],
            self.params["input2_c"],
            self.params["input2_w"],
            self.params["input2_h"],
        )
        self._add_input(
            self.params["input3_idx"],
            self.params["input3_dtype"],
            self.params["input3_c"],
            self.params["input3_w"],
            self.params["input3_h"],
        )
        self._add_input(
            self.params["input4_idx"],
            self.params["input4_dtype"],
            self.params["input4_c"],
            self.params["input4_w"],
            self.params["input4_h"],
        )
        self._add_output(
            self.params["output_idx"],
            self.params["output_dtype"],
            self.params["output_c"],
            self.params["output_w"],
            self.params["output_h"],
        )

        if None in default_params4:
            warnings.warn(f"parameters are not all set for op {self.params['op']}")

    def get_macs(self) -> int:
        return 0

    def generate_inference_str(self):
        string = ""
        params = self.params
        string += (
            f"mconcat4({str(int(params['input_h']*params['input_w']))}, "
            + f"{self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
            + f"{self._getBufferstr(params['input2_buf_add'], params['input2_buf_add_offset'])},"
            + f"{self._getBufferstr(params['input3_buf_add'], params['input3_buf_add_offset'])},"
            + f"{self._getBufferstr(params['input4_buf_add'], params['input4_buf_add_offset'])},"
            + f"{params['input_c']},{params['input2_c']},{params['input3_c']},{params['input4_c']},"
            + f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])});\n"
        )
        return string