import numpy as np
import os
import os.path as osp
import tqdm
import torch 
import sys
import logging
from torch import optim
import torch.nn.functional as func
import random
from datetime import datetime


from models.TrafficStream import Basic_Model as TrafficStream 
from models.Linear import MLP, Linear, NLinear, DLinear
from models.PatchTST import PatchTST
from models.informer import informer
from models.TrafficStream import graph_constructor 
from utils import common_tools as ct
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np
from models import detect
from models import replay
from models.ewc import EWC


from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import to_dense_batch, k_hop_subgraph
from torch_geometric.data import Data, Dataset
import networkx as nx

from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils.timefeatures import time_features



result = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}},  96:{"mae":{}, "mape":{}, "rmse":{}}}
pin_memory = True 
n_work = 2

def seed_set(seed=0):
    max_seed = (1 << 32) - 1
    random.seed(seed)
    np.random.seed(random.randint(0, max_seed))
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True



def z_score(data):
    return (data - np.mean(data)) / np.std(data)

class PEMS3_Stream: #* 从PEMS raw data中生成训练验证测试数据集 + graph
    def __init__(self, args, savepath, train_rate=0.6, val_rate=0.2, test_rate=0.2, val_test_mix=False):
        self.args = args 
        raw_data = np.load(osp.join(args.raw_data_path, str(args.phase)+".npz"))["x"]
        
        self.data = raw_data[0:args.days*288, :]
        # self.data = z_score(self.data) #! 这里对原代码做了修改，直接对整个数据集做归一化
        t, n = self.data.shape[0], self.data.shape[1]
        
        train_idx = [i for i in range(int(t*train_rate))]
        val_idx = [i for i in range(int(t*train_rate), int(t*(train_rate+val_rate)))]
        test_idx = [i for i in range(int(t*(train_rate+val_rate)), t)]

        # print(train_idx, val_idx, test_idx)
        
        train_x, train_y = self.slice_dataset(train_idx)
        val_x, val_y = self.slice_dataset(val_idx)
        test_x, test_y = self.slice_dataset(test_idx)

        if val_test_mix: #* 和常规setting不同
            val_test_x = np.concatenate((val_x, test_x), 0)
            val_test_y = np.concatenate((val_y, test_y), 0)
            val_test_idx = np.arange(val_x.shape[0]+test_x.shape[0])
            np.random.shuffle(val_test_idx)
            val_x, val_y = val_test_x[val_test_idx[:int(t*val_rate)]], val_test_y[val_test_idx[:int(t*val_rate)]]
            test_x, test_y = val_test_x[val_test_idx[int(t*val_rate):]], val_test_y[val_test_idx[int(t*val_rate):]]

        train_x = z_score(train_x)
        val_x = z_score(val_x)
        test_x = z_score(test_x)
        graph = nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(args.phase)+"_adj.npz"))["x"])
        args.graph_size = graph.number_of_nodes() #! wait to be used 
        edge_index = np.array(list(graph.edges)).T   #* PEMS data 包含2部分，一部分是graph，一部分是sensor data 
        del graph

        np.savez(savepath, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, edge_index=edge_index)
        # prepared_data = {"train_x":train_x, "train_y":train_y, "val_x":val_x, "val_y":val_y, "test_x":test_x, "test_y":test_y, "edge_index":edge_index}
        # return prepared_data
    
    def slice_dataset(self, idx, x_len=12, y_len=12):
        x = self.args.x_len 
        y = self.args.y_len 
        res = self.data[idx]
        node_size = self.data.shape[1]
        t = len(idx)-1
        idic = 0
        x_index, y_index = [], []
        
        for i in tqdm.tqdm(range(t,0,-1)):
            if i-x_len-y_len>=0:
                x_index.extend(list(range(i-x_len-y_len, i-y_len)))
                y_index.extend(list(range(i-y_len, i)))

        x_index = np.asarray(x_index)
        y_index = np.asarray(y_index)
        x = res[x_index].reshape((-1, x_len, node_size))
        y = res[y_index].reshape((-1, y_len, node_size))
                        
        return x, y

class TrafficDataset(Dataset): #? 意义？ 仅仅是作为dataloader的中介吗？
    def __init__(self, nodes, inputs, split, x='', y='', edge_index=''): #* mode means subgraph or not
            self.x = inputs[split+'_x'][:, :, nodes] # [T, Len, N]
            self.y = inputs[split+'_y'][:, :, nodes] # [T, Len, N]
            self.x_mask = inputs[split+'_x_mask']
            self.y_mask = inputs[split+'_y_mask']
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T) #* exchange the dimension of Time and Node 
        y = torch.Tensor(self.y[index].T)
        return Data(x=x, y=y), torch.Tensor(self.x_mask[index]), torch.Tensor(self.y_mask[index])  #* the input form is a graph, x and y seen as attribute of each node

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0,0,0]
        border2s = [len(df_raw),len(df_raw),len(df_raw)]
        border1 = border1s[0]
        border2 = border2s[0]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        self.seq_x = []
        self.seq_y = []
        self.seq_x_mark = []
        self.seq_y_mark = []
        for index in range(len(self)): 
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            self.seq_x.append(self.data_x[s_begin:s_end])
            self.seq_y.append(self.data_y[r_begin:r_end])
            self.seq_x_mark.append(self.data_stamp[s_begin:s_end])
            self.seq_y_mark.append(self.data_stamp[r_begin:r_end])
        self.seq_x = np.array(self.seq_x)
        self.seq_y = np.array(self.seq_y)
        self.seq_x_mark = np.array(self.seq_x_mark)
        self.seq_y_mark = np.array(self.seq_y_mark)


    def __getitem__(self, index):
        return self.seq_x[index], self.seq_y[index], self.seq_x_mark[index], self.seq_y_mark[index]


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 0,0]
        border2s = [len(df_raw),len(df_raw),len(df_raw)]
        border1 = border1s[0]
        border2 = border2s[0]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        self.seq_x = []
        self.seq_y = []
        self.seq_x_mark = []
        self.seq_y_mark = []
        for index in range(len(self)): 
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            self.seq_x.append(self.data_x[s_begin:s_end])
            self.seq_y.append(self.data_y[r_begin:r_end])
            self.seq_x_mark.append(self.data_stamp[s_begin:s_end])
            self.seq_y_mark.append(self.data_stamp[r_begin:r_end])
        self.seq_x = np.array(self.seq_x)
        self.seq_y = np.array(self.seq_y)
        self.seq_x_mark = np.array(self.seq_x_mark)
        self.seq_y_mark = np.array(self.seq_y_mark)

    def __getitem__(self, index):
        return self.seq_x[index], self.seq_y[index], self.seq_x_mark[index], self.seq_y_mark[index]


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, 0, 0]
        border2s = [len(df_raw), len(df_raw), len(df_raw)]
        border1 = border1s[0]
        border2 = border2s[0]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        self.seq_x = []
        self.seq_y = []
        self.seq_x_mark = []
        self.seq_y_mark = []
        for index in range(len(self)): 
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            self.seq_x.append(self.data_x[s_begin:s_end])
            self.seq_y.append(self.data_y[r_begin:r_end])
            self.seq_x_mark.append(self.data_stamp[s_begin:s_end])
            self.seq_y_mark.append(self.data_stamp[r_begin:r_end])

        self.seq_x = np.array(self.seq_x)
        self.seq_y = np.array(self.seq_y)
        self.seq_x_mark = np.array(self.seq_x_mark)
        self.seq_y_mark = np.array(self.seq_y_mark)



    def __getitem__(self, index):
        return self.seq_x[index], self.seq_y[index], self.seq_x_mark[index], self.seq_y_mark[index]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def get_dataset(args):
    inputs = None   
    
    if args.data_name == 'PEMS3-Stream': 
        if args.data_process:
            PEMS3_Stream(args, osp.join(args.save_data_path, str(args.phase)+'_30day'), val_test_mix=True)
        inputs = np.load(osp.join(args.save_data_path, str(args.phase)+"_30day.npz"), allow_pickle=True)
    else: 
        args.save_data_path = "data/" + args.data_name + "/"
        args.data_path = './data/' + args.data_name + '.csv'

        data_label = "Custom"
        if args.data_name == 'ETTh2':data_label = "ETT_hour" 
        if args.data_name == 'ETTm1':data_label = "ETT_minute" 
    
        if args.data_process:
            timeenc = 0 if args.embed!='timeF' else 1
            freq=args.freq
            ds = eval("Dataset_" + data_label)(root_path=args.root_path,
                data_path=args.data_path,
                size=[args.x_len, args.label_len, args.y_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq)
            if args.phase_len < 1: args.phase_len = int(len(ds) * args.phase_len) 
            t = 0; tmp_phase = 1
            while t + args.phase_len < len(ds):
                train_x, train_y = ds[t:t+args.phase_len][0], ds[t:t+args.phase_len][1] 
                train_x_mask, train_y_mask = ds[t:t+args.phase_len][2], ds[t:t+args.phase_len][3]
                maxtest = min(t+args.phase_len*2, len(ds)) 
                mintest = t + args.phase_len 
                val_idx = np.random.choice(np.arange(mintest, maxtest), int((maxtest-mintest)*args.val_ratio), replace=False)
                test_idx = []
                for i in range(mintest, maxtest): 
                    if i in val_idx: continue
                    test_idx.append(i)
                val_x, val_y = ds[val_idx][0], ds[val_idx][1]
                val_x_mask, val_y_mask = ds[val_idx][2], ds[val_idx][3]
                test_x, test_y = ds[test_idx][0], ds[test_idx][1]
                test_x_mask, test_y_mask = ds[test_idx][2], ds[test_idx][3]
                # detect dir exist or not, if not, create it
                if not osp.exists(osp.join(args.save_data_path, str(args.phase_len))):
                    os.makedirs(osp.join(args.save_data_path, str(args.phase_len)))

                np.savez(osp.join(args.save_data_path, str(args.phase_len), str(tmp_phase)), train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, train_x_mask=train_x_mask, train_y_mask = train_y_mask, val_x_mask=val_x_mask, val_y_mask=val_y_mask, test_x_mask= test_x_mask, test_y_mask=test_y_mask)
                t += args.phase_len
                tmp_phase += 1
            args.data_process = False 
                
        inputs = np.load(osp.join(args.save_data_path,  str(args.phase_len), str(args.phase)+".npz"), allow_pickle=True)
        args.graph_size = inputs['train_x'].shape[-1]
        args.subgraph = torch.arange(args.graph_size)
    return inputs 

def init_args(args): 
    static_args = {
        ##* static args
        "load_config": "configs/", 
        "data_process": False ,
        "auto_test": 1,
        "load": True,
        "device": "cuda:0",
        "build_graph": False,
        "dynamic_graph": False, 
        "graph_input": True, 
        
        ##* dataset related args
        "model_name": "TrafficStream",
        "data_name": "PEMS3-Stream",
        "root_path": "",
        "raw_data_path": "data/district3F11T17/finaldata/",
        "graph_path": "data/district3F11T17/graph/",
        "save_data_path": "data/district3F11T17/FastData/",
        "model_path": "exp/district3F11T17/",
        "phase": 2012,
        "days": 31,        
    }

    import json

    ##* load model related args 
    with open(static_args["load_config"] + args.conf + ".json", "r") as f:
        config = json.load(f)
    for key, value in config.items():
        static_args[key] = value


    # merge static_args into args, the key is the attribute of args, the value is the value of args
    if args is None:
        class Args:
            pass
        args = Args()
    
    for key, value in static_args.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    args.logname = args.conf
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    args.path = osp.join(args.model_path, args.logname+args.time)
    ct.mkdirs(args.path)
    # global result
    # result[args.y_len] = {"mae":{}, "mape":{}, "rmse":{}}
    return args

class base_framework: 
    def __init__(self, args) -> None:
        self.args = args 

    def load_best_model(self):
        load_path = osp.join(self.args.model_path, self.args.logname+self.args.time, str(self.args.phase-1), "best_model.pkl")
        ##!! file name file name file name 
    
        self.args.logger.info("[*] load from {}".format(load_path))
        state_dict = torch.load(load_path, map_location=self.args.device)["model_state_dict"]
        if 'tcn2.weight' in state_dict:
            del state_dict['tcn2.weight']
            del state_dict['tcn2.bias']

        #* model + load_state_dict
        

        model = eval(self.args.model_name)(self.args) #TODO   modify model name
        model.load_state_dict(state_dict)
        

        model = model.to(self.args.device)
        self.model = model 

    def prepare(self): 
        ##* 1.init graph 
        self.args.gc = None    ##todo 改到前面来， increment strategy不需要 gc 
        if args.graph_input == True:
            init_graph(self.args)             
            self.args.subgraph = torch.arange(self.args.graph_size) 
        ##* 2.prep complete data (form as follow ) 
        self.inputs = get_dataset(self.args) 
        # to show: print(inputs["train_x"].shape, inputs["train_y"].shape, inputs["val_x"].shape, inputs["val_y"].shape, inputs["test_x"].shape, inputs["test_y"].shape, inputs["edge_index"].shape)
        self.args.logger.info("[*] phase " + str(self.args.phase) + " Dataset load!")
         
        ##* 3.apply certain strategy (like get select nodes and construct subgraph)
        if self.inc_state:
            self.incremental_strategy()
        else:
            self.static_strategy()

        if self.args.build_graph: 
            self.args.gc = graph_constructor(len(self.args.subgraph), self.args.build_subgraph_size, self.args.node_emb_dim, self.args.device, alpha=self.args.tanhalpha, static_feat=None).to(self.args.device)
            self.args.idx = torch.arange(len(self.args.subgraph)).to(self.args.device)

        ##* prep dl 
        if self.args.train:
            self.train_loader = DataLoader(TrafficDataset(self.args.subgraph.numpy(), self.inputs, "train"), batch_size=self.args.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=n_work,drop_last=True)
            self.val_loader = DataLoader(TrafficDataset(self.args.subgraph.numpy(), self.inputs,"val"), batch_size=self.args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work,drop_last=False)
        self.test_loader = DataLoader(TrafficDataset(np.arange(self.args.graph_size), self.inputs,"test"), batch_size=self.args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work,drop_last=False)
        x=0
        

    def incremental_strategy(self):  #! incremental prepare work 目的是获得增量节点及其对应的subgraph
        
        if self.args.subgraph_train:
            node_list = list()
            ##* increase nodes
            if self.args.increase:
                cur_node_size = np.load(osp.join(self.args.graph_path, str(self.args.phase)+"_adj.npz"))["x"].shape[0]
                pre_node_size = np.load(osp.join(self.args.graph_path, str(self.args.phase-1)+"_adj.npz"))["x"].shape[0] 
                node_list.extend(list(range(pre_node_size, cur_node_size)))
                pass

            ##* influence nodes
            if self.args.detect: 
                self.args.logger.info("[*] detect strategy {}".format(self.args.detect_strategy))
                pre_data = np.load(osp.join(self.args.raw_data_path, str(self.args.phase-1)+".npz"))["x"]
                cur_data = np.load(osp.join(self.args.raw_data_path, str(self.args.phase)+".npz"))["x"]
                pre_graph = np.array(list(nx.from_numpy_matrix(np.load(osp.join(self.args.graph_path, str(self.args.phase-1)+"_adj.npz"))["x"]).edges)).T
                cur_graph = np.array(list(nx.from_numpy_matrix(np.load(osp.join(self.args.graph_path, str(self.args.phase)+"_adj.npz"))["x"]).edges)).T
                # 20% of current graph size will be sampled
                self.args.topk = int(0.01*self.args.graph_size) 
                influence_node_list = detect.influence_node_selection(self.model, self.args, pre_data, cur_data, pre_graph, cur_graph)
                node_list.extend(list(influence_node_list))
                pass

            ##* replay nodes
            if self.args.replay: 
                self.args.replay_num_samples = int(0.09*self.args.graph_size) #int(0.2*self.args.graph_size)- len(node_list)
                self.args.logger.info("[*] replay node number {}".format(self.args.replay_num_samples))
                replay_node_list = replay.replay_node_selection(self.args, self.inputs, self.model)
                node_list.extend(list(replay_node_list))
                pass
            
            #* sample nodes 
            node_list = list(set(node_list))
            if len(node_list) > int(0.1*self.args.graph_size):
                node_list = random.sample(node_list, int(0.1*self.args.graph_size)) #*超过就采样 
                #! 是从 increase_node + replay_node + influence_node 中随机选取 10% 的节点 
            
            
            # Obtain subgraph of node list
            cur_graph = torch.LongTensor(np.array(list(nx.from_numpy_matrix(np.load(osp.join(self.args.graph_path, str(self.args.phase)+"_adj.npz"))["x"]).edges)).T)
            edge_list = list(nx.from_numpy_matrix(np.load(osp.join(self.args.graph_path, str(self.args.phase)+"_adj.npz"))["x"]).edges)
            graph_node_from_edge = set()

            for (u,v) in edge_list:
                graph_node_from_edge.add(u)
                graph_node_from_edge.add(v)
            node_list = list(set(node_list) & graph_node_from_edge)  #! 排除孤立点 
                
           
            if len(node_list) != 0 :
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_list, num_hops=self.args.num_hops, edge_index=cur_graph, relabel_nodes=True)
                self.args.subgraph = subgraph  #! may induce error
                self.args.subgraph_edge_index = subgraph_edge_index
                self.args.mapping = mapping

            self.args.logger.info("number of increase nodes:{}, nodes after {} hop:{}, total nodes this phase {}".format\
                        (len(node_list), self.args.num_hops, self.args.subgraph.size(), self.args.graph_size))
            self.args.node_list = np.asarray(node_list)

            graph = nx.Graph()
            graph.add_nodes_from(range(self.args.subgraph.size(0)))
            graph.add_edges_from(self.args.subgraph_edge_index.numpy().T)
            adj = nx.to_numpy_array(graph)
            adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
            self.args.sub_adj = torch.from_numpy(adj).to(torch.float).to(self.args.device) 
        else :  ##* no subgraph train
            if self.args.graph_input:
                self.args.subgraph = torch.arange(self.args.graph_size) 
                self.args.sub_adj = self.args.adj   

        if self.args.load: 
            self.load_best_model() 

        if self.args.ewc:
            self.args.logger.info("[*] EWC! lambda {:.6f}".format(self.args.ewc_lambda))
            self.model = EWC(self.model, self.args.adj, self.args.ewc_lambda, self.args.ewc_strategy)
            ewc_loader = DataLoader(TrafficDataset(np.arange(0, self.args.graph_size), self.inputs, "train"), batch_size=self.args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
            self.model.register_ewc_params(ewc_loader, func.mse_loss, self.args.device)
            pass
        
    
    def static_strategy(self):    
        if self.args.graph_input == True: 
            self.args.subgraph = torch.arange(self.args.graph_size) 
            self.args.sub_adj = self.args.adj   

        if self.args.phase == self.args.begin_phase: 
            self.model = eval(self.args.model_name)(self.args).to(self.args.device)
        if self.args.strategy == "static" and self.args.phase > self.args.begin_phase: 
            self.args.train = False 
        if self.args.strategy == "retrain": 
            self.model = eval(self.args.model_name)(self.args).to(self.args.device)


    def train(self): 
        global result
        path = osp.join(self.args.path, str(self.args.phase))
        ct.mkdirs(path)

        ##* Model Optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        if self.args.loss == "mse": lossfunc = func.mse_loss
        elif self.args.loss == "huber": lossfunc = func.smooth_l1_loss


        ##* train start
        self.args.logger.info("[*] phase " + str(self.args.phase) + " Training start")
        global_train_steps = len(self.train_loader) // self.args.batch_size +1

        iters = len(self.train_loader)
        lowest_validation_loss = 1e7
        counter = 0
        patience = 5
        self.model.train()
        use_time = []
        validation_loss_list = []
        for epoch in range(self.args.epoch): #* train body 
            training_loss = 0.0
            start_time = datetime.now()
            loss2=0#*
            # Train Model
            cn = 0
            for batch_idx, (data, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
                if epoch == 0 and batch_idx == 0:
                    self.args.logger.info("node number {}".format(data.x.shape))
                data = data.to(self.args.device, non_blocking=pin_memory)

                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)

                optimizer.zero_grad()
                tmp_y = data.y.reshape(-1, self.args.graph_size, self.args.y_len+self.args.label_len)
                dec_inp = torch.zeros(tmp_y.shape[0], self.args.y_len, self.args.graph_size).float().to(self.args.device)
                tmp_y = tmp_y.permute(0, 2, 1)
                dec_inp = torch.cat([tmp_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)
                

                if self.args.graph_input: 
                    pred = self.model(data, self.args.sub_adj)
                else : 
                    if self.args.linear_output:
                        pred = self.model(data)
                    else: 
                        pred = self.model(data, batch_x_mark, dec_inp, batch_y_mark)
            
                if self.args.subgraph_train and self.inc_state:
                    pred, _ = to_dense_batch(pred, batch=data.batch)
                    data.y, _ = to_dense_batch(data.y, batch=data.batch) 
                    pred = pred[:, self.args.mapping, :]
                    data.y = data.y[:, self.args.mapping, :] 

                # print(data.y.shape, pred.shape)
                pred = pred[:, -self.args.y_len:]
                data.y = data.y[:, -self.args.y_len:]

                loss = lossfunc(data.y, pred, reduction="mean")
                loss2 = masked_mae_np(data.y.cpu().data.numpy(), pred.cpu().data.numpy(), 0) 
                if self.args.ewc and self.inc_state:
                    loss += self.model.compute_consolidation_loss()
                training_loss += float(loss2)
                loss.backward()
                optimizer.step()
                
                cn += 1

            if epoch == 0:
                total_time = (datetime.now() - start_time).total_seconds()
            else:
                total_time += (datetime.now() - start_time).total_seconds()
            use_time.append((datetime.now() - start_time).total_seconds())
            training_loss = training_loss/cn 
    
            # Validate Model
            validation_loss = 0.0
            cn = 0
            with torch.no_grad():
                for batch_idx, (data, batch_x_mark, batch_y_mark) in enumerate(self.val_loader):
                    data = data.to(self.args.device, non_blocking=pin_memory)
                    batch_x_mark = batch_x_mark.float().to(self.args.device)
                    batch_y_mark = batch_y_mark.float().to(self.args.device)

                    tmp_y = data.y.reshape(-1, self.args.graph_size, self.args.y_len+self.args.label_len)
                    dec_inp = torch.zeros(tmp_y.shape[0], self.args.y_len, self.args.graph_size).float().to(self.args.device)
                    tmp_y = tmp_y.permute(0, 2, 1)
                    dec_inp = torch.cat([tmp_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)
                    

                    if self.args.graph_input: 
                        pred = self.model(data, self.args.sub_adj)
                    else : 
                        if self.args.linear_output:
                            pred = self.model(data)
                        else: 
                            pred = self.model(data, batch_x_mark, dec_inp, batch_y_mark)
                    if self.args.subgraph_train and self.inc_state:
                        pred, _ = to_dense_batch(pred, batch=data.batch)
                        data.y, _ = to_dense_batch(data.y, batch=data.batch)
                        pred = pred[:, self.args.mapping, :]
                        data.y = data.y[:, self.args.mapping, :]
                    pred = pred[:, -self.args.y_len:]
                    data.y = data.y[:, -self.args.y_len:]
                    loss = masked_mae_np(data.y.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                    # print("eval: ",loss)
                    validation_loss += float(loss)
                    cn += 1
            validation_loss = float(validation_loss/cn)
            validation_loss_list.append(validation_loss)

            self.args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")

            # Early Stop
            if validation_loss <= lowest_validation_loss:
                counter = 0
                lowest_validation_loss = round(validation_loss, 4)
                save_model = self.model
                if self.inc_state and self.args.ewc:
                    save_model = self.model.model 
                torch.save({'model_state_dict': save_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+("_epoch_%d.pkl" % epoch)))
            else:
                counter += 1
                if counter > patience:
                    break
        pass
        epoch_idx = np.argmin(validation_loss_list)
        best_model_path = osp.join(path, str(lowest_validation_loss)+("_epoch_%d.pkl" % epoch_idx))
        
        best_model = eval(self.args.model_name)(self.args)
        best_model.load_state_dict(torch.load(best_model_path, self.args.device)["model_state_dict"])
        torch.save({'model_state_dict': best_model.state_dict()}, osp.join(path, "best_model.pkl"))
        self.model = best_model
        self.model = self.model.to(self.args.device)        
        
        # result[self.args.phase]["total_time"] =  total_time
        # result[self.args.phase]["average_time"] =  sum(use_time)/len(use_time)
        # result[self.args.phase]["epoch_num"] =  epoch+1
        # self.args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))
    
    def report_result(self):
        global result
        for i in [3, 6, 12, self.args.y_len]:
            for j in ['mae', 'rmse', 'mape']:
                info = ""
                for phase in range(self.args.begin_phase, self.args.end_phase+1):
                    if i in result:
                        if j in result[i]:
                            if phase in result[i][j]:
                                info+="{:.2f}\t".format(result[i][j][phase])
                self.args.logger.info("{}\t{}\t".format(i,j) + info)

        for phase in range(self.args.begin_phase, self.args.end_phase+1):
            if phase in result:
                info = "phase\t{}\ttotal_time\t{}\taverage_time\t{}\tepoch\t{}".format(phase, result[phase]["total_time"], result[phase]["average_time"], result[phase]['epoch_num'])
                self.args.logger.info(info)

    def test_model(self):
        ##! 
        self.args.idx = torch.arange(self.args.graph_size).to(self.args.device, non_blocking=pin_memory)

        self.model.eval()
        pred_ = []
        truth_ = []
        loss = 0.0
        with torch.no_grad():
            cn = 0
            for data,  batch_x_mark, batch_y_mark in self.test_loader:
                data = data.to(self.args.device, non_blocking=pin_memory)
                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)

                tmp_y = data.y.reshape(-1, self.args.graph_size, self.args.y_len+self.args.label_len)
                dec_inp = torch.zeros(tmp_y.shape[0], self.args.y_len, self.args.graph_size).float().to(self.args.device)
                tmp_y = tmp_y.permute(0, 2, 1)
                dec_inp = torch.cat([tmp_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)
                
                if self.args.graph_input: 
                    pred = self.model(data, self.args.sub_adj)
                else : 
                    if self.args.linear_output:
                        pred = self.model(data)
                    else: 
                        pred = self.model(data, batch_x_mark, dec_inp, batch_y_mark)
                pred = pred[:, -self.args.y_len:]
                data.y = data.y[:, -self.args.y_len:]
                loss += func.mse_loss(data.y, pred, reduction="mean")
                pred, _ = to_dense_batch(pred, batch=data.batch)
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred_.append(pred.cpu().data.numpy())
                truth_.append(data.y.cpu().data.numpy())
                cn += 1
            loss = loss/cn
            self.args.logger.info("[*] loss:{:.4f}".format(loss))
            pred_ = np.concatenate(pred_, 0)
            truth_ = np.concatenate(truth_, 0)
            mae = base_framework.metric(truth_, pred_, self.args)
            return loss
        
    @staticmethod
    def metric(ground_truth, prediction, args):
        global result
        pred_time = [3,6,12, args.y_len]
        args.logger.info("[*] phase {}, testing".format(args.phase))
        for i in pred_time:
            mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
            rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
            mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
            args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
            # print(result[i],i)            
            result[i]["mae"][args.phase] = mae
            result[i]["mape"][args.phase] = mape
            result[i]["rmse"][args.phase] = rmse
        return mae

    def run(self): 
        ##* logger、dir prep
        logger = init_log(self.args)  
        logger.info("params : %s", vars(self.args))
        ct.mkdirs(self.args.save_data_path)

        ##* multi-phase train 
        self.inc_state = False 
        for phase in range(self.args.begin_phase, self.args.end_phase): ##todo 将第一年拿出来单独处理 
            self.args.phase = phase 

            self.args.logger.info("[*] phase {} load from {}_30day.npz".format(self.args.phase, osp.join(self.args.save_data_path, str(phase)))) 
            
            self.prepare()
            if self.args.train:
                self.train()
            else: 
                state_dict = torch.load(self.args.test_model_path, map_location=self.args.device)["model_state_dict"]
                self.model.load_state_dict(state_dict)

            self.test_model()

            if self.args.strategy == "incremental": self.inc_state = True 

        self.report_result()

def init_graph(args):
    if args.build_graph == False:  ##TODO 这里要分类讨论吗 ？ 
        adj = np.load(osp.join(args.graph_path, str(args.phase)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        args.adj = torch.from_numpy(adj).to(torch.float).to(args.device) #* adj -> normalized 邻接矩阵
        args.graph_size = adj.shape[0]
    else : 
        adj = np.load(osp.join(args.graph_path, str(args.phase)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        args.adj = torch.from_numpy(adj).to(torch.float).to(args.device) #* adj -> normalized 邻接矩阵
        args.graph_size = adj.shape[0]
        # adj = np.load(osp.join(args.graph_path, str(args.phase)+"_adj.npz"))["x"]
        # args.adj = None 
        # args.graph_size = adj.shape[0]


def init_log(args):
    log_dir, log_filename = args.path, args.logname
    logger = logging.getLogger(__name__)
    ct.mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(log_dir, log_filename+".log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join(log_dir, log_filename+".log"))
    args.logger = logger
    return logger

def main(args):
    args = init_args(args) ##* init args
    fm = base_framework(args)
    fm.run()
        
    

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="incremental-naive")
    parser.add_argument("--data_name", type=str, default="electricity")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args() ##* parse args
    # args = None  
    seed_set(13) ##* set seed
    main(args) ##* run any framework for a time 
