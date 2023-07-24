from typing import Any
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import numpy as np

class Enhancer(object):
    def __init__(self, args) -> None:
        self.args = args 

    def identity(self, x): 
        return x 

    def jitter(self, x, sigma=0.08):
        # https://arxiv.org/pdf/1706.00527.pdf
        return x + torch.randn(size=x.shape).to(self.args.device) * sigma

    def spike(self, x): 
        x = x.permute(0, 2, 1)
        idx = torch.randint(low=0, high=self.args.seq_len-1, size=(x.shape[0], x.shape[1], 1), device=x.device) 
        amplify = torch.randint(low=5, high=20, size=(x.shape[0], x.shape[1], 1), device=x.device) 
        sign = torch.sign(torch.randn( size=(x.shape[0], x.shape[1], 1), device=x.device)) 
        max_value = torch.max(x, dim=-1, keepdim=True)[0] 
        # print(max_value, max_value.shape)
        # convert every number in idx to a three dimentional vector, first two dimention is the index of the number, the third dimention is the number itself
        first_axis = torch.arange(x.shape[0], device=x.device).reshape(x.shape[0], 1, 1).repeat(1, x.shape[1], 1)
        second_axis = torch.arange(x.shape[1], device=x.device).reshape(1, x.shape[1], 1).repeat(x.shape[0], 1, 1)

        # idx = torch.cat([first_axis, second_axis, idx], dim=-1).reshape(-1, 3)
        x[first_axis, second_axis, idx] += sign * max_value * amplify 
        return x.permute(0, 2, 1) 

    def substitude(self, x): 
        if x.shape[2] == 1: return x
        x = x.permute(0, 2, 1)
        b = x.shape[0] 
        for i in range(len(b)): 
            c_idx = torch.randint(low=0, high=x.shape[1] - 1, size=(x.shape[1],), device=x.device) 
            t_idx = torch.randint(low=0, high=x.shape[2] - 24, size=(1,), device=x.device) 
            x[i, :, t_idx:t_idx+24] = x[i, c_idx, t_idx:t_idx+24]

        return x.permute(0, 2, 1)
    
    def __call__(self, x) -> Any:
        bs = x.shape[0] // 4
        x = torch.cat([x[:bs], self.jitter(x[bs:bs*2]), self.spike(x[bs*2:bs*3]), self.substitude(x[bs*3:])], dim=0)
        return x