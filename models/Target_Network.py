import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, args, base_model):
        super(Model, self).__init__()
        self.base_model = base_model
        #todo 

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.base_model(x)
        return x # to [Batch, Output length, Channel]
