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
        args.cm = None
        if not args.same_init:
            self.base_model = eval(args.model_name).Model(args)
        else : 
            self.base_model = args.Base_T
        # if hasattr(self.base_model, "model"): self.base_model.model.cm = None
        # if hasattr(self.base_model, "configs"): self.base_model.configs.cm = None 
        self.args = args 
        print(self.args.refiner, self.args.use_cm,self.args.cm)

        #todo 

    def forward(self, x, *args, feature=False, given_feature = None):
        # x: [Batch, Input length, Channel]
        self.args.use_cm = False
        # print("now T start")
        # print(self.args.use_cm)
        x, F = self.base_model(x, *args, given_feature = given_feature)
        self.args.use_cm = True
        # print("now T end")
        
        if not feature:
            return x
        return x, F # to [Batch, Output length, Channel]
