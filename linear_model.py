import torch
from torch.types import Tensor
from typing import TypeVar

T = TypeVar("T", bound="LinearModel")


class LinearModel:
    def __init__(self, r_local: Tensor, coefficients: Tensor):
        self.r_local: Tensor = r_local
        self.coefficients: Tensor = coefficients

    @classmethod
    def fit(cls: type[T], x: Tensor, y: Tensor) -> T:
        r_local, beta = cls.fit_n(x, y)
        return cls(r_local, beta)

    @classmethod
    def fit_n(cls, x: Tensor, y: Tensor):
        r_xy_or_xy: Tensor = torch.cat([x, y], dim=1)
        return cls.ols_n(r_xy_or_xy)

    @classmethod
    def ols_n(cls, r_xy_or_xy: Tensor) -> tuple[Tensor, Tensor]:
        _, r_s = torch.linalg.qr(r_xy_or_xy)

        r = r_s[:-1, :-1].clone()
        theta = r_s[:-1, -1:].clone()

        b: Tensor = torch.linalg.solve_triangular(r, theta, upper=True)

        return (r_s, b)

    @classmethod
    def update_distributed(cls: type[T], lm: T, r_remote: Tensor) -> T:
        r_local, beta = cls.update_distributed_n(lm.r_local, r_remote)
        return LinearModel(r_local, beta)

    @classmethod
    def update_distributed_n(cls, r_local, r_remote) -> tuple[Tensor, Tensor]:
        return cls.ols_n(torch.cat([r_local, r_remote], dim=0))

    @classmethod
    def update(cls: type[T], r_local: Tensor, x: Tensor, y: Tensor) -> T:
        r_local, beta = LinearModel.update_n(r_local, x, y)
        return cls(r_local, beta)

    @classmethod
    def update_n(cls, r_local: Tensor, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        return cls.ols_n(torch.cat([r_local, torch.cat([x, y], dim=1)]))
