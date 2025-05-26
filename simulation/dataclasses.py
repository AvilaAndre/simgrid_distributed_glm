from dataclasses import dataclass
from torch.types import Tensor
from linear_model import LinearModel

import sys


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
            + sys.getsizeof(self.sender)
            + sys.getsizeof(self.flow)
            + sys.getsizeof(self.estimate)
        )


@dataclass
class LMState:
    model: LinearModel
    r_remotes: dict[str, Tensor]
    nodes: list[str]
