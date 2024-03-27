import torch
from torch import nn
from torch import Tensor
from torch.nn import Transformer
import math
from typing import Iterable, List
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

