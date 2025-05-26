from dataclasses import dataclass
from torch.types import Tensor

@dataclass
class CoefficientsMsg:
    coefficients: Tensor
