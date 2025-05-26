import torch
from torch.types import Tensor
from enum import Enum

class FamilyEnum(Enum):
    GAUSSIAN = 1
    BINOMIAL = 2

class Gaussian:
    def linkinv(eta: Tensor) -> Tensor:
        return eta

    def mu_eta(eta: Tensor) -> Tensor:
        return torch.ones_like(eta)

    def variance(mu: Tensor) -> Tensor:
        return torch.ones_like(mu)


class Binomial:
    def linkinv(eta: Tensor) -> Tensor:
        exp = torch.exp(eta)
        return exp / (1 + exp)

    def mu_eta(eta: Tensor) -> Tensor:
        op = 1 + torch.exp(eta)
        return torch.exp(eta) / (op * op)

    def variance(mu: Tensor) -> Tensor:
        return mu * (1 - mu)

