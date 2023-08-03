import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import numpy as np
from models import DLinear
from models import PatchTST 
from models import informer
from models.layers.PatchTST_backbone import _MultiheadAttention

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
        dropout = 0.2,
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
            nn.Linear(dim, dim)
            
        ) #*

        self.shrinkage = Shrinkage(dim, gap_size=(1)) #todo
        self.add_residual = add_residual
        self.self_attn = _MultiheadAttention(dim, 16, 64, 64, attn_dropout=0, proj_dropout=0, res_attention=False)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):
        

        out, _ = self.self_attn(x, x, x, key_padding_mask=None, attn_mask=None)
        # out = self.shrinkage(out.permute(0, 2, 1)).permute(0, 2, 1) #* soft threshold before residual
        out  = self.dropout_attn(out)
        out = self.to_out(out)

        if self.add_residual:
            out = out + x

        return out

class Refiner(nn.Module): 
    def __init__(self, args):
        super(Refiner, self).__init__()
        self.args = args 
        self.refiner_block_num = args.refiner_block_num
        self.blocks = nn.ModuleList([
            Refiner_block(args.d_model, add_residual= (not args.refiner_no_residual)) for _ in range(self.refiner_block_num)
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
        self.Aligner = nn.Linear(args.d_model, args.d_model)
        self.Refiner = nn.Linear(args.d_model, args.d_model) # simplest implementation of Refiner
        # self.Refiner = nn.Linear(args.d_model, args.d_model) # simplest implementation of Refiner
        # self.Refiner = nn.Sequential(*[nn.Linear(args.d_model, args.d_model), nn.ReLU(), nn.Linear(args.d_model, args.d_model)])
        # self.Refiner = Refiner(args)
        
        self.args = args 

    def forward(self, x):
        # x: [Batch, C，d, P ]
        # x = self.Refiner(x)
        if len(x.shape) == 3: x = x.unsqueeze(1)
        x_ = x.permute(0, 1, 3, 2)
        #todo x also be refined
        x_refined = x 
        # print(self.args.refiner)
        if self.args.refiner:
            # print(x_.shape)
            x_refined = self.Refiner(x_).permute(0, 1, 3, 2)
        if self.args.share_head:  #* use the T forecastor after aligne
            x_refined = self.Aligner(x_refined.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        return x_refined.squeeze(), self.Aligner(x_).permute(0, 1, 3, 2).squeeze()

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.correction_module = Correction_Module(args)
        args.cm = self.correction_module
        self.base_model = eval(args.model_name).Model(args) # cm 通过args嵌入到模型内部
        self.args = args
        #todo 

    def forward(self, x, *args, feature=False, given_feature=None):
        # x: [Batch, Input length, Channel]
        if not self.args.share_head: #* controling use the F from Aligner 
            x, F = self.base_model(x, *args, given_feature=given_feature)
        else: 
            x, F = self.base_model(x, *args, given_feature=given_feature)
            x, _ = self.base_model(x, *args, given_feature=F)
        # print(x_.shape) #* 16 * 1 * 128 * 42
        # print(x.shape) #* 16 * 96 * 1
        # x = self.correction_module(x)
        if not feature:
            return x
        return x, F # to [Batch, Output length, Channel]
