import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import DLinear
from models import PatchTST 
from models import informer
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.base_model = eval(args.model_name).Model(args)
        #todo 

    def forward(self, x, feature=False):
        # x: [Batch, Input length, Channel]
        x, F = self.base_model(x)
        if not feature:
            return x
        return x, F # to [Batch, Output length, Channel]
