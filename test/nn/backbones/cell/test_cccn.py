"""Unit tests for CCCN"""

import torch
from ...._utils.nn_module_auto_test import NNModuleAutoTest
from topobenchmark.nn.backbones.cell.cccn import CCCN


def test_cccn(random_graph_input):
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    auto_test = NNModuleAutoTest([
        {
            "module" : CCCN, 
            "init": (x.shape[1], ),
            "forward": (x, edges_1, edges_2),
            "assert_shape": x.shape
        }
    ])