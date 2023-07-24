import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import numpy as np
from models import DLinear
from models import PatchTST 
from models import informer

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)
    
class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # average = torch.mean(x, dim=1, keepdim=True)  #CS
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x

class Refiner_block(nn.Module):
    def __init__(
        self,
        dim,
        query_key_dim = 32, ##todo
        expansion_factor = 2.,
        add_residual = True, ##todo
        causal = False,
        dropout = 0.,
        laplace_attn_fn = False,
        rel_pos_bias = False,
        norm_klass = nn.LayerNorm
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim) 

        self.norm = norm_klass(dim)
        self.dropout = nn.Dropout(dropout) 
        
        self.attn_fn = ReLUSquared()

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.offsetscale = OffsetScale(query_key_dim, heads = 2)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.shrinkage = Shrinkage(dim, gap_size=(1)) #todo
        self.add_residual = add_residual


    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):
        seq_len, device = x.shape[-2], x.device

        normed_x = self.norm(x)
        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1)
        
        qk = self.to_qk(normed_x)
        q, k = self.offsetscale(qk)

        sim = einsum('b i d, b j d -> b i j', q, k)

        attn = self.attn_fn(sim / seq_len)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = out * gate

        out = self.to_out(out)

        out = self.shrinkage(out.permute(0, 2, 1)).permute(0, 2, 1) #* soft threshold before residual

        if self.add_residual:
            out = out + x

        return out

class Refiner(nn.Module): 
    def __init__(self, args):
        super(Refiner, self).__init__()
        self.args = args 
        self.refiner_block_num = 1 
        self.blocks = nn.ModuleList([
            Refiner_block(args.d_model) for _ in range(self.refiner_block_num)
        ])

    def forward(self, x):
        # x: [Batch, C，P, d ]
        tmp = x.shape[0]
        x = x.reshape(-1, x.shape[-2], x.shape[-1]) 

        for i in range(self.refiner_block_num):
            x = self.blocks[i](x)
        
        x = x.reshape(tmp, -1, x.shape[-2], x.shape[-1]) 
        return x   


class Correction_Module(nn.Module):
    def __init__(self, args):
        super(Correction_Module, self).__init__()
        self.label = np.zeros(args.batch_size)
        self.Aligner = nn.Linear(args.d_model, args.d_model)
        # self.Refiner = nn.Linear(args.d_model, args.d_model) # simplest implementation of Refiner
        self.Refiner = Refiner(args)
        self.args = args 

    def forward(self, x):
        # x: [Batch, C，d, P ]
        # x = self.Refiner(x)
        x_ = x.permute(0, 1, 3, 2)
        #todo x also be refined
        x_refined = x 
        # print(self.args.refiner)
        if self.args.refiner:
            x_refined = self.Refiner(x_).permute(0, 1, 3, 2)

        return x_refined, self.Aligner(x_)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.correction_module = Correction_Module(args)
        args.cm = self.correction_module
        self.base_model = eval(args.model_name).Model(args) # cm 通过args嵌入到模型内部
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
