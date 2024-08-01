import warnings

from ..constant import USE_BIT_MASK
from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

__all__ = ["Gather"]

default_params = {
    # op related
    "op": "Gather",
    "input_idx": None,
# "input2_idx": None,
    "output_idx": None,
    # tensor related
    "input_dim": None,
    "input_h": None,
    "input_w": None,
    "input_c": None,
# "input2_c": None,
    "output_dim": None,
    "output_h": None,
    "output_w": None,
    "output_c": None,
    "input_dtype": "int8",
# "input2_dtype": "int32",
    "output_dtype": "int8",
    "poses":None,
}


class Gather(basicOperator):
    idx=0
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
        # self._add_input(
        #     self.params["input2_idx"],
        #     self.params["input2_dtype"],
        #     self.params["input2_c"],
        # )
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
        poses=[str(x) for x in params['poses']]
        poses_str=",".join(poses)
        string +=f"static int32_t temp_{Gather.idx:02d}[{len(params['poses'])}]="+"{"+poses_str+"};\n"

        string += (
            f"mgather({str(int(params['input_h']*params['input_w']))}, "
            + f"{self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
            + f"temp_{Gather.idx:02d},"
            + f"{params['input_c']},{len(params['poses'])},"
            + f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])});\n"
        )
        Gather.idx += 1
        return string
