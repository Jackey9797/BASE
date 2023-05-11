import math
import pdb
import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F

class BatchGCNConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True, gcn=True):
        super(BatchGCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_neigh = nn.Linear(in_features, out_features, bias=bias)
        if not gcn:
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter('weight_self', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.weight_neigh.reset_parameters()
        if self.weight_self is not None:
            self.weight_self.reset_parameters()

    def forward(self, x, adj):
        # x: [bs, N, in_features], adj: [N, N]
        input_x = torch.matmul(adj, x)              # [N, N] * [bs, N, in_features] = [bs, N, in_features]
        output = self.weight_neigh(input_x)             # [bs, N, in_features] * [in_features, out_features] = [bs, N, out_features]
        if self.weight_self is not None:
            output += self.weight_self(x)               # [bs, N, out_features]
        return output

class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat 

    #! 怎么做embedding的，用一段序列见图还是 
    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj

class Basic_Model(nn.Module):
    """Some Information about Basic_Model"""
    def __init__(self, args):
        super(Basic_Model, self).__init__()
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

        self.args = args

    def forward(self, data, adj, gc=None):
        if self.args.build_graph and self.args.gc != None:
            adj = self.args.gc(self.args.idx)   
        N = adj.shape[0]
        
        if self.args.dynamic_graph: 
            x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
            # caculate correlation matrix
            def corrcoef(x):
                x_reducemean = x - torch.mean(x, dim=-1, keepdim=True)
                numerator = torch.matmul(x_reducemean, x_reducemean.permute(0,2,1))
                no = torch.norm(x_reducemean, dim=-1).unsqueeze(1)
                denominator = torch.matmul(no.permute(0,2,1),no)
                corrcoef = numerator / (denominator + 1e-8)
                return corrcoef
            adj_d = corrcoef(x[:10])
            adj_d = torch.abs(adj_d)
            mask = torch.zeros(10,N, N).to(self.args.device)
            mask.fill_(float('0'))
            s1,t1 = adj_d.topk(self.args.DG_k,2)
            mask.scatter_(2,t1,s1.fill_(1))
            adj_d = adj_d*mask
            adj_d = torch.mean(adj_d, dim=0)

            if self.args.DG_type == "add":
                adj = (adj + adj_d) / 2
            if self.args.DG_type == "mul":
                adj = adj * adj_d

            del adj_d, mask, s1, t1
            # keep topk elements of the last 2 dimension of adj_d, and set others to 0

        # print("check on number of N" + str(N))
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        ##
        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x

        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

    def feature(self, data, adj, gc=None): #* used for pattern detection
        if self.args.build_graph and self.args.gc != None:
            adj = self.args.gc(self.args.idx)   
        N = adj.shape[0]

        if self.args.dynamic_graph: 
            x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
            # caculate correlation matrix
            def corrcoef(x):
                x_reducemean = x - torch.mean(x, dim=-1, keepdim=True)
                numerator = torch.matmul(x_reducemean, x_reducemean.permute(0,2,1))
                no = torch.norm(x_reducemean, dim=-1).unsqueeze(1)
                denominator = torch.matmul(no.permute(0,2,1),no)
                corrcoef = numerator / (denominator + 1e-8)
                return corrcoef
            adj_d = corrcoef(x[:10])
            adj_d = torch.abs(adj_d)
            mask = torch.zeros(10,N, N).to(self.args.device)
            mask.fill_(float('0'))
            s1,t1 = adj_d.topk(self.args.DG_k,2)
            mask.scatter_(2,t1,s1.fill_(1))
            adj_d = adj_d*mask
            adj_d = torch.mean(adj_d, dim=0)

            if self.args.DG_type == "add":
                adj = (adj + adj_d) / 2
            if self.args.DG_type == "mul":
                adj = adj * adj_d

        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]
        ##
        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        ##
        return x