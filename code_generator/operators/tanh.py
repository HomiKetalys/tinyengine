import warnings

from ..constant import USE_BIT_MASK
from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

__all__ = ["Tanh"]

default_params = {
    # op related
    "op": "Tanh",
    "input_idx": None,
    "output_idx": None,
    # tensor related
    "input_dim": None,
    "input_h": None,
    "input_w": None,
    "input_c": None,
    "output_dim": None,
    "output_h": None,
    "output_w": None,
    "output_c": None,
    "input_dtype": "int8",
    "output_dtype": "int8",
    # trainable parameters
    "input_zero_point": None,
    "output_zero_point": None,
    "input_scale": None,
    "output_scale": None,
}


class Tanh(basicOperator):
    idx = 0

    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params)
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
        self._add_output(
            self.params["output_idx"],
            self.params["output_dtype"],
            self.params["output_c"],
            self.params["output_w"],
            self.params["output_h"],
        )

        if None in default_params:
            warnings.warn(f"parameters are not all set for op {self.params['op']}")

    def get_macs(self) -> int:
        return 0

    def generate_inference_str(self):
        string = ""
        params = self.params

        string += (
                f"mtanh({str(int(params['input_h'] * params['input_w'] * params['input_c']))}, "
                + f"{self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
                + f"{str(params['input_scale'])},{str(params['input_zero_point'])},"
                + f"{str(params['output_scale'])},{str(params['output_zero_point'])},"
                + f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])});\n"
        )
        return string
