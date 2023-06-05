import numpy as np
import os
import os.path as osp
import tqdm
import torch 
import json
import sys
import logging
from torch import optim
import torch.nn.functional as func
import random
from datetime import datetime


from utils import common_tools as ct


from torch.utils.data import DataLoader, dataloader, Dataset

from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils.timefeatures import time_features


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

def process_data_stream(args):
    data_label = "Custom"
    if args.data_name == 'ETTh2': data_label = "ETT_hour" 
    if args.data_name == 'ETTm1': data_label = "ETT_minute" 

    timeenc = 0 if args.embed!='timeF' else 1
    freq=args.freq
    ds = eval("Dataset_" + data_label)(root_path=args.root_path,
        data_path=args.data_path,
        size=[args.x_len, args.label_len, args.y_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq)
    args.phase_len = int(len(ds) * args.phase_len_ratio) 
    t = 0; tmp_phase = 1
    while t + args.phase_len < len(ds):
        train_x, train_y = ds[t:t+args.phase_len][0], ds[t:t+args.phase_len][1] 
        train_x_mask, train_y_mask = ds[t:t+args.phase_len][2], ds[t:t+args.phase_len][3]
        maxtest = min(t+args.phase_len*2, len(ds)) 
        mintest = t + args.phase_len 
        if args.val_test_mix:
            val_idx = np.random.choice(np.arange(mintest, maxtest), int((maxtest-mintest)*args.val_ratio), replace=False)
        else : val_idx = np.arange(mintest, maxtest)[:int((maxtest-mintest)*args.val_ratio)]
        test_idx = []
        for i in range(mintest, maxtest): 
            if i in val_idx: continue
            test_idx.append(i)
        val_x, val_y = ds[val_idx][0], ds[val_idx][1]
        val_x_mask, val_y_mask = ds[val_idx][2], ds[val_idx][3]
        test_x, test_y = ds[test_idx][0], ds[test_idx][1]
        test_x_mask, test_y_mask = ds[test_idx][2], ds[test_idx][3]
        # detect dir exist or not, if not, create it
        if not osp.exists(osp.join(args.save_data_path, "_".join([str(args.phase_len_ratio),  str(args.x_len), str(args.y_len)]))):
            os.makedirs(osp.join(args.save_data_path, "_".join([str(args.phase_len_ratio),  str(args.x_len), str(args.y_len)])))

        np.savez(osp.join(args.save_data_path, "_".join([str(args.phase_len_ratio),  str(args.x_len), str(args.y_len)]),str(tmp_phase)), train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, train_x_mask=train_x_mask, train_y_mask = train_y_mask, val_x_mask=val_x_mask, val_y_mask=val_y_mask, test_x_mask= test_x_mask, test_y_mask=test_y_mask)
        t += args.phase_len
        tmp_phase += 1

def get_dataset(args):
    inputs = None   

    args.save_data_path = "data/processed_data/" + args.data_name + "/"
    args.data_path = './data/' + args.data_name + '.csv'

    if args.data_process:
        process_data_stream(args)
        args.data_process = False 
            
    inputs = np.load(osp.join(args.save_data_path, "_".join([str(args.phase_len_ratio), str(args.x_len), str(args.y_len)]), str(args.phase)+".npz"), allow_pickle=True)
    args.nodes = torch.arange(inputs['train_x'].shape[-1])
    args.enc_in = len(args.nodes)

    return inputs 

