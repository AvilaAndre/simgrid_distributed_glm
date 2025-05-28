from dataclasses import dataclass
from torch.types import Tensor
from linear_model import LinearModel
from generalized_linear_model import GeneralizedLinearModel

import sys


@dataclass
class LMState:
    model: LinearModel
    r_remotes: dict[str, Tensor]
    nodes: list[str]


@dataclass
class ModelCoefficients:
    coefficients: Tensor


@dataclass
class LMConcatMessage:
    origin: str
    r_remote: Tensor

    def size(self) -> int:
        return (
            sys.getsizeof(self)
            + sys.getsizeof(self.origin)
            + sys.getsizeof(self.r_remote)
        )


@dataclass
class GLMState:
    model: GeneralizedLinearModel
    data: tuple[Tensor, Tensor]
    r_remotes: dict[str, int]
    total_nrow: int
    nodes: list[str]
    finished: bool


@dataclass
class GLMSumRowsMessage:
    origin: str
    nrows: int

    def size(self) -> int:
        return (
            sys.getsizeof(self) + sys.getsizeof(self.origin) + sys.getsizeof(self.nrows)
        )


@dataclass
class GLMConcatMessage:
    origin: str
    r_remote: Tensor
    iter: int

    def size(self) -> int:
        return (
            sys.getsizeof(self)
            + sys.getsizeof(self.origin)
            + sys.getsizeof(self.r_remote)
            + sys.getsizeof(self.iter)
        )
