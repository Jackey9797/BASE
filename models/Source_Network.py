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

class ReconHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """


        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        return x

class Ref_block(nn.Module):
    def __init__(
        self,
        args,
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
        self.args = args

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
            nn.Linear(dim, dim // 2),
            nn.Linear(dim // 2, dim)
        ) #*
        # self.FFN = nn.Linear(dim, dim)
        
        self.shrinkage = Shrinkage(dim, gap_size=(1)) #todo
        self.add_residual = args.add_residual

        self.self_attn = _MultiheadAttention(dim, 16, args.mid_dim, args.mid_dim, attn_dropout=0, proj_dropout=0, res_attention=False)

        # Add & Norm
        self.dropout_attn = nn.Dropout(self.args.ref_dropout)
        if self.args.add_FFN: 
            self.FFN = nn.Sequential(*[nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)])

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):
        
        key_mask = mask
        out_1, _ = self.self_attn(x, x, x, key_padding_mask=key_mask, attn_mask=None) 
        
        if self.args.ref_dropout > 0:
            out_1  = self.dropout_attn(out_1)

        #* todo FFN
        if self.args.add_FFN: 
            out_1 = self.FFN(out_1)

        if self.add_residual:
            out_1 = out_1 + x

        return out_1

class Rec_block(nn.Module):
    def __init__(
        self,
        args,
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
        self.args = args

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
            nn.Linear(dim, dim // 2),
            nn.Linear(dim // 2, dim)
        ) #*
        # self.FFN = nn.Linear(dim, dim)
        
        self.shrinkage = Shrinkage(dim, gap_size=(1)) #todo
        self.add_residual = add_residual
        self.self_attn = _MultiheadAttention(dim, 16, args.mid_dim, args.mid_dim, attn_dropout=0, proj_dropout=0, res_attention=False)

        # Add & Norm

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):
        attn_mask = torch.tril(torch.ones((x.shape[-2],x.shape[-2])), self.args.mask_border) * torch.triu(torch.ones((x.shape[-2],x.shape[-2])), -self.args.mask_border)
        # print(x.shape, attn_mask.shape)
        out_1, _ = self.self_attn(x, x, x, key_padding_mask=None, attn_mask=attn_mask.to(x.device))
        
        rec_score = torch.mean((out_1 - x) ** 2, dim=[2])
        # print(out_1[4].median(), x[4].median())
        q = torch.tensor([0.25, 0.5, 0.75], device=out_1.device) 
        q1, q2, q3 = torch.quantile(rec_score, q, dim=-1) 
        tmp = out_1 - x 

        if self.args.train == 0 and self.args.debugger == 1: 
            print(rec_score[2].shape, rec_score[2], (q3 + self.args.theta * (q3 - q1))[2])
            self.args.show = self.args.show.reshape(rec_score.shape + (8,))
            self.args.show = self.args.show * (rec_score < (q3 + self.args.theta * (q3 - q1)).unsqueeze(-1)).unsqueeze(-1)
        # def vis(idx, channel): 
        
        out_1 = x + tmp

        if self.add_residual:
            out = out_1 + x

        return out_1

class Refiner(nn.Module): 
    def __init__(self, args):
        super(Refiner, self).__init__()
        self.args = args 
        self.ref_block_num = args.ref_block_num
        # self.ref_block_num = 2  #todo 
        self.rec_block_num = 1
        self.refiner_residual = False
        self.rec_residual = False
        # self.ref = nn.Sequential(*[
        #     Refiner_block(args, args.d_model, add_residual= self.refiner_residual) for _ in range(self.refiner_block_num)
        # ])
        self.ref = nn.ModuleList([Ref_block(args, args.d_model, add_residual= self.rec_residual) for _ in range(self.ref_block_num)])
        self.rec = nn.Sequential(*[Rec_block(args, args.d_model, add_residual= self.rec_residual) for _ in range(self.rec_block_num)])
        self.rec_head = ReconHead(args.d_model, 8, 0.1)

    def forward(self, x):
        # x: [Batch, C，P, d ]
        tmp = x.shape[0]
        x = x.reshape(-1, x.shape[-2], x.shape[-1]).detach()  #! detach here
        # print(x.requires_grad)
        # if rec == False: # in P branch
        for i in self.rec.parameters():
            i.requires_grad = False
        # print("lst", x.shape)
        
        r = self.rec(x) 
        # print(r.requires_grad)
        

        rec_score = torch.mean((r - x) ** 2, dim=[2])
        q = torch.tensor([0.25, 0.5, 0.75], device=r.device) 
        q1, q2, q3 = torch.quantile(rec_score, q, dim=-1) 
        rec_score = (rec_score > (q3 + self.args.theta * (q3 - q1)).unsqueeze(-1)) # [890 42]
        self.args.rs_before = torch.mean(torch.mean((r - x) ** 2, dim=[2]) ) 

        #* --- reconstruct ---
        x_ = x * (~rec_score).unsqueeze(-1)
        for i in range(self.ref_block_num): 
            x_ = self.ref[i](x_, mask=rec_score) #? Do we need mask? 
        #* --- reconstruct ---
        # print(x_.requires_grad)
        
        
        # set all idx of rec_sc of x to x_'s value todo  #* 两种实现 #todo 
        #* 中间加正则限制 

        if not self.args.rec_all:
            x_[rec_score == 0] = x[rec_score == 0]  #* keep normal patch still
        x_ = torch.cat([x_[:, :-int(x.shape[1] * (1 - self.args.rec_length_ratio)), :], x[:, -int(x.shape[1] * (1 - self.args.rec_length_ratio)):, :]], dim=1)   #* keep last half
        
        r = self.rec(x_)   #? whehter use rec score as parameters

        # if not self.args.rec_all:
        self.args.rs_after = torch.mean(torch.mean((r - x_) ** 2, dim=[2])) 

        for i in self.rec.parameters():
            i.requires_grad = True
        if self.args.train == 0: print("aha")
        x_ = x_.reshape(tmp, -1, x_.shape[-2], x_.shape[-1]) 
        return x_
        
           


class Correction_Module(nn.Module):
    def __init__(self, args):
        super(Correction_Module, self).__init__()
        # self.Aligner = nn.Linear(args.d_model, args.d_model)
        self.Aligner = nn.Identity()
        # self.Refiner = nn.Linear(args.d_model, args.d_model) # simplest implementation of Refiner
        # self.Refiner = nn.Linear(args.d_model, args.d_model) # simplest implementation of Refiner
        # self.Refiner = nn.Sequential(*[nn.Linear(args.d_model, args.d_model), nn.ReLU(), nn.Linear(args.d_model, args.d_model)])
        self.Refiner = Refiner(args)
        
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
            x_refined = self.Refiner(x_)
            # x_refined = self.Refiner(x_refined)
            x_refined = x_refined.permute(0, 1, 3, 2)
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
