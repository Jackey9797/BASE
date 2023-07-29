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

    def jitter(self, x, sigma=0.2):
        # use the plot to visualize x[3]
        # import matplotlib.pyplot as plt

        # plt.plot(x[3].cpu().numpy()) 
        # plt.savefig('before.png')
        # plt.close()

        x = x + torch.randn(size=x.shape).to(self.args.device) * sigma
        # plt.plot(x[3].cpu().numpy()) 
        # plt.savefig('after.png')
        return x

    def spike(self, x): 
        x = x.permute(0, 2, 1)

        # import matplotlib.pyplot as plt
        # print(x[3].shape)
        # plt.plot(x[3][1].cpu().numpy()) 
        # plt.savefig('before1.png')
        # plt.close()
        
        idx = torch.randint(low=0, high=self.args.seq_len-1, size=(x.shape[0], x.shape[1], 1), device=x.device) 
        amplify = torch.randint(low=4, high=15, size=(x.shape[0], x.shape[1], 1), device=x.device) #todo
        sign = torch.sign(torch.randn( size=(x.shape[0], x.shape[1], 1), device=x.device)) 
        max_value = torch.max(x, dim=-1, keepdim=True)[0] 
        # print(max_value, max_value.shape)
        # convert every number in idx to a three dimentional vector, first two dimention is the index of the number, the third dimention is the number itself
        first_axis = torch.arange(x.shape[0], device=x.device).reshape(x.shape[0], 1, 1).repeat(1, x.shape[1], 1)
        second_axis = torch.arange(x.shape[1], device=x.device).reshape(1, x.shape[1], 1).repeat(x.shape[0], 1, 1)

        # idx = torch.cat([first_axis, second_axis, idx], dim=-1).reshape(-1, 3)
        x[first_axis, second_axis, idx] += sign * max_value * amplify 
        # plt.plot(x[3][1].cpu().numpy()) 
        # plt.savefig('after1.png')
        # plt.close()
        
        return x.permute(0, 2, 1) 

    def substitude(self, x): 
        if x.shape[2] == 1: return x
        x = x.permute(0, 2, 1)
        # import matplotlib.pyplot as plt
        # plt.plot(x[3][1].cpu().numpy()) 
        # plt.savefig('before2.png')
        # plt.close()
        b = x.shape[0] 
        for i in range(b): 
            c_idx = torch.randint(low=0, high=x.shape[1] - 1, size=(x.shape[1],), device=x.device) 
            t_idx = torch.randint(low=0, high=x.shape[2] - 48, size=(1,), device=x.device) 
            x[i, :, t_idx:t_idx+48] = x[i, c_idx, t_idx:t_idx+48] #todo
        # plt.plot(x[3][1].cpu().numpy()) 
        # plt.savefig('after2.png')
        # plt.close()
        # exit(0)
        return x.permute(0, 2, 1)
    
    def __call__(self, x) -> Any:
        bs = x.shape[0] // 4
        if self.args.enhance_type == 1: 
            x = torch.cat([x[:bs], self.jitter(x[bs:])], dim=0)
        elif self.args.enhance_type == 2:     
            x = torch.cat([x[:bs], self.spike(x[bs:])], dim=0)
        elif self.args.enhance_type == 3: 
            x = torch.cat([x[:bs], self.substitude(x[bs:])], dim=0)
        else:
            x = torch.cat([x[:bs], self.jitter(x[bs:bs*2]), self.spike(x[bs*2:bs*3]), self.substitude(x[bs*3:])], dim=0)
        return x