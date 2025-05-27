import unittest
import torch

from simulation.glm import GeneralizedLinearModel
from simulation.lm import LinearModel
from family import FamilyEnum


dtype = torch.float64


class TestGeneralizedLinearModel(unittest.TestCase):
    def test_list_int(self):
        """
        with family=Gaussian is equal to Linear Model
        """
        y_1 = torch.tensor([[57], [55], [2]], dtype=dtype)
        x_1 = torch.tensor([[15], [4], [13]], dtype=dtype)
        x_1 = torch.cat([torch.ones_like(x_1), x_1], dim=1)

        y_2 = torch.tensor([[5], [50], [25]], dtype=dtype)
        x_2 = torch.tensor([[3], [9], [1]], dtype=dtype)
        x_2 = torch.cat([torch.ones_like(x_2), x_2], dim=1)

        x = torch.cat([x_1, x_2], dim=0)
        y = torch.cat([y_1, y_2], dim=0)

        lm = LinearModel.fit(x, y)
        glm = GeneralizedLinearModel.fit(x, y, {"family": FamilyEnum.GAUSSIAN})

        self.assertTrue(torch.allclose(lm.coefficients, glm.coefficients))

    def test_binomial_family(self):
        x, y = TestGeneralizedLinearModel.gen_data_binomial()
        glm = GeneralizedLinearModel.fit(x, y, {"family": FamilyEnum.BINOMIAL})

        beta_hat = torch.tensor([[-18.796313], [1.843344]], dtype=dtype)

        self.assertTrue(torch.allclose(glm.coefficients, beta_hat))

    def gen_data_binomial():
        y = torch.tensor([1] * 40 + [0] * 40, dtype=torch.float64).reshape(80, 1)

        x_vals = [
            13.192708,
            14.448090,
            9.649417,
            10.320116,
            11.647111,
            13.650080,
            14.084886,
            12.802728,
            12.112105,
            12.037028,
            13.965425,
            15.977136,
            12.453174,
            15.544261,
            11.953416,
            13.113029,
            9.641135,
            12.180368,
            12.420320,
            11.397851,
            12.907001,
            11.365693,
            10.844742,
            14.534238,
            15.422245,
            13.413604,
            12.944656,
            9.860943,
            11.712985,
            16.808641,
            14.413471,
            13.190056,
            10.360630,
            10.401697,
            13.787254,
            10.377768,
            13.894956,
            10.628888,
            14.807852,
            13.198363,
            5.837473,
            10.179199,
            6.625756,
            3.322105,
            9.080811,
            3.816459,
            6.389127,
            4.613805,
            7.655697,
            5.843509,
            7.382162,
            8.495226,
            8.414558,
            9.663929,
            9.625316,
            5.727101,
            5.629926,
            7.671450,
            7.717170,
            9.424363,
            6.922032,
            8.777796,
            9.283351,
            8.233282,
            6.498535,
            4.674278,
            9.026015,
            10.186052,
            8.900973,
            3.947789,
            11.527118,
            7.360485,
            7.442523,
            7.757168,
            9.294623,
            6.753300,
            10.380745,
            5.958523,
            6.278818,
            10.985235,
        ]
        x = torch.tensor(x_vals, dtype=torch.float64).reshape(80, 1)
        x = torch.cat([torch.ones_like(x), x], dim=1)

        return x, y
