import json
import logging
import os
import unittest

import numpy as np


class TestPrepare(unittest.TestCase):
    def test():
        input_dir = "memmap_dataset"
        # Load metadata
        # test precision
        with open(os.path.join(input_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        loaded_data = {}

        for split_name, info in metadata["splits"].items():
            memmap_path = os.path.join(input_dir, info["subdir"], info["filename"])
            if not os.path.exists(memmap_path):
                logging.error(f"{memmap_path} for {split_name} not found")
                continue
            size = tuple(info["shape"])[0]
            data = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=tuple(info["shape"]))
            if np.any(data[:, :4] <= 0):
                op_zero = len(np.where(data[:, 0] <= 0))
                hi_zero = len(np.where(data[:, 1] <= 0))
                lo_zero = len(np.where(data[:, 2] <= 0))
                cl_zero = len(np.where(data[:, 3] <= 0))
                print(
                    f"Data contains non-positve values in is_train '{split_name}', open: {op_zero}, high: {hi_zero}, low: {lo_zero}, close: {cl_zero}"
                )
            try:
                data[size - 1, 0]
            except IndexError:
                logging.error(f"IndexError: {split_name} {size}")
                assert False

        return loaded_data
