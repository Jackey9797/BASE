import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(MLP, self).__init__()
        self.x_len = configs.x_len
        self.y_len = configs.y_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.graph_size
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Sequential(*[nn.Linear(self.x_len, 64), nn.ReLU(), nn.Linear(64, self.y_len)]))
        else:
            self.Linear = nn.Sequential(*[nn.Linear(self.x_len, 64),  nn.ReLU(), nn.Linear(64, self.y_len)])

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = x.x.reshape((-1,self.channels ,self.x_len)).permute(0, 2, 1)
        if self.individual:
            output = torch.zeros([x.size(0),self.y_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            # print(x.shape)
        # print(x.shape)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        return x.reshape(-1, x.shape[2]) # [Batch, Output length, Channel]