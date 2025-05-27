import unittest
import torch

from simulation.LM import LinearModel


dtype = torch.float64


class TestLinearModel(unittest.TestCase):
    def test_update_equals_fit_with_all_values(self):
        y_1 = torch.tensor([[57], [55], [2]], dtype=dtype)
        x_1 = torch.tensor([[15], [4], [13]], dtype=dtype)
        x_1 = torch.cat([torch.ones_like(x_1), x_1], dim=1)

        y_2 = torch.tensor([[5], [50], [25]], dtype=dtype)
        x_2 = torch.tensor([[3], [9], [1]], dtype=dtype)
        x_2 = torch.cat([torch.ones_like(x_2), x_2], dim=1)

        lm1 = LinearModel.fit(torch.cat([x_1, x_2]), torch.cat([y_1, y_2]))
        lm2 = LinearModel.fit(x_1, y_1)

        lm2 = LinearModel.update(lm2.r_local, x_2, y_2)

        self.assertTrue(torch.allclose(lm1.coefficients, lm2.coefficients))

    def test_update_distributed_equals_fit_with_all_values(self):
        y_1 = torch.tensor([[57], [55], [2]], dtype=dtype)
        x_1 = torch.tensor([[15], [4], [13]], dtype=dtype)
        x_1 = torch.cat([torch.ones_like(x_1), x_1], dim=1)

        y_2 = torch.tensor([[5], [50], [25]], dtype=dtype)
        x_2 = torch.tensor([[3], [9], [1]], dtype=dtype)
        x_2 = torch.cat([torch.ones_like(x_2), x_2], dim=1)

        lm1 = LinearModel.fit(torch.cat([x_1, x_2]), torch.cat([y_1, y_2]))
        lm2 = LinearModel.fit(x_1, y_1)
        lm3 = LinearModel.fit(x_2, y_2)

        lm3 = LinearModel.update_distributed(lm3, lm2.r_local)

        self.assertTrue(torch.allclose(lm1.coefficients, lm3.coefficients))
