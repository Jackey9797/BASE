import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import DLinear
from models import PatchTST 
from models import informer
class Correction_Module(nn.Module):
    def __init__(self, args):
        super(Correction_Module, self).__init__()
        self.label = np.zeros(args.batch_size)
        self.Aligner = nn.Linear(args.d_model, args.d_model)
        self.Refiner = nn.Linear(args.d_model, args.d_model) # simplest implementation of Refiner
        self.args = args 

    def forward(self, x):
        # x: [Batch, C，d, P ]
        # x = self.Refiner(x)
        x_ = x.permute(0, 1, 3, 2)
        #todo x also be refined
        return x, self.Aligner(x_)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.correction_module = Correction_Module(args)
        args.cm = self.correction_module
        self.base_model = eval(args.model_name).Model(args)
        #todo 

    def forward(self, x, feature=False):
        # x: [Batch, Input length, Channel]
        x, F = self.base_model(x)
        # print(x_.shape) #* 16 * 1 * 128 * 42
        # print(x.shape) #* 16 * 96 * 1
        # x = self.correction_module(x)
        if not feature:
            return x
        return x, F # to [Batch, Output length, Channel]
