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


from models.TrafficStream import Basic_Model as TrafficStream 
from models.Linear import MLP, Linear, NLinear, DLinear
from models.PatchTST import PatchTST
from models.informer import informer
from models.TrafficStream import graph_constructor 
from utils import common_tools as ct
from utils.tools import visualize
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np
from models import detect
from models import replay
from models.ewc import EWC

from sklearn.preprocessing import StandardScaler
import pandas as pd

from data_process import get_dataset
from torch.utils.data import DataLoader, dataloader, Dataset




result = {}
pin_memory = True 
n_work = 4

def seed_set(seed=0):
    max_seed = (1 << 32) - 1
    random.seed(seed)
    np.random.seed(random.randint(0, max_seed))
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True

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

class TSDataset(Dataset):
    def __init__(self, nodes, inputs, split, x='', y='', edge_index=''): #* mode means nodes or not
            self.x = inputs[split+'_x'][:, :, nodes] # [T, Len, N]
            self.y = inputs[split+'_y'][:, :, nodes] # [T, Len, N]
            self.x_mask = inputs[split+'_x_mask']
            self.y_mask = inputs[split+'_y_mask']
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index])  
        y = torch.Tensor(self.y[index])
        return x, y, torch.Tensor(self.x_mask[index]), torch.Tensor(self.y_mask[index])  #* the input form is a graph, x and y seen as attribute of each node

class base_framework: 
    def __init__(self, args) -> None:
        self.args = args 

    def load_best_model(self):
        load_path = osp.join(self.args.exp_path, self.args.logname+self.args.time, str(self.args.phase-1), "best_model.pkl")
        state_dict = torch.load(load_path, map_location=self.args.device)["model_state_dict"]
        model = eval(self.args.model_name)(self.args) 
        self.args.logger.info("[*] load from {}".format(load_path))
        model.load_state_dict(state_dict)
        model = model.to(self.args.device)
        self.model = model 

    def prepare(self): 
        self.inputs = get_dataset(self.args) 
        # to show: print(inputs["train_x"].shape, inputs["train_y"].shape, inputs["val_x"].shape, inputs["val_y"].shape, inputs["test_x"].shape, inputs["test_y"].shape, inputs["edge_index"].shape)
        self.args.logger.info("[*] phase " + str(self.args.phase) + " Dataset load!")
        
        #* apply strategy 
        if self.inc_state:
            self.incremental_strategy()
        else:
            self.static_strategy()

        # if self.args.build_graph: 
        #     self.args.gc = graph_constructor(len(self.args.nodes), self.args.build_subgraph_size, self.args.node_emb_dim, self.args.device, alpha=self.args.tanhalpha, static_feat=None).to(self.args.device)
        #     self.args.idx = torch.arange(len(self.args.nodes)).to(self.args.device)

        ##* prep dl 
        if self.args.train:
            self.train_loader = DataLoader(TSDataset(self.args.nodes.numpy(), self.inputs, "train"), batch_size=self.args.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=n_work,drop_last=True)
            self.val_loader = DataLoader(TSDataset(self.args.nodes.numpy(), self.inputs,"val"), batch_size=self.args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work,drop_last=False)
        self.test_loader = DataLoader(TSDataset(np.arange(self.args.enc_in), self.inputs,"test"), batch_size=self.args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work,drop_last=False)

    def incremental_strategy(self):  
        if self.args.load: 
            self.load_best_model() 
    
    def static_strategy(self):    
        self.model = eval(self.args.model_name)(self.args).to(self.args.device)
        global result
        result[self.args.y_len] = {"mae":{}, "mape":{}, "rmse":{}}

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
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
                # data.x[:, 48:48+24] = 0
                batch_x = batch_x.float().to(self.args.device, non_blocking=pin_memory)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)

                optimizer.zero_grad()
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)

                if self.args.graph_input: 
                    pred = self.model(batch_x, self.args.sub_adj)
                else : 
                    if self.args.linear_output:
                        pred = self.model(batch_x)
                    else: 
                        pred = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = pred[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

                pred = outputs
                true = batch_y

                loss = lossfunc(pred, true, reduction="mean")

                # if self.args.ewc and self.inc_state:
                #     loss += self.model.compute_consolidation_loss()
                training_loss += float(loss)
                loss.backward()
                optimizer.step()
                
                cn += 1
                # if cn % 10 == 1: 
                #     visualize(data.cpu(), data.y.cpu(), pred.cpu())

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
                for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.val_loader):
                    batch_x = batch_x.float().to(self.args.device, non_blocking=pin_memory)
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float().to(self.args.device)
                    batch_y_mark = batch_y_mark.float().to(self.args.device)

                    optimizer.zero_grad()
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)

                    if self.args.graph_input: 
                        pred = self.model(batch_x, self.args.sub_adj)
                    else : 
                        if self.args.linear_output:
                            pred = self.model(batch_x)
                        else: 
                            pred = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                

                    # print(data.y.shape, pred.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = pred[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
                    
                    loss = lossfunc(pred, true, reduction="mean")
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
        for i in [self.args.y_len]:
            for j in ['mae', 'rmse', 'mape']:
                info = ""
                for phase in range(self.args.begin_phase, self.args.end_phase+1):
                    if i in result:
                        if j in result[i]:
                            if phase in result[i][j]:
                                info+="{:.4f}\t".format(result[i][j][phase])
                self.args.logger.info("{}\t{}\t".format(i,j) + info)

        # for phase in range(self.args.begin_phase, self.args.end_phase+1):
        #     if phase in result:
        #         info = "phase\t{}\ttotal_time\t{}\taverage_time\t{}\tepoch\t{}".format(phase, result[phase]["total_time"], result[phase]["average_time"], result[phase]['epoch_num'])
        #         self.args.logger.info(info)

    def test_model(self):
        ##! 
        self.args.idx = torch.arange(self.args.enc_in).to(self.args.device, non_blocking=pin_memory)

        self.model.eval()
        pred_ = []
        truth_ = []
        loss = 0.0
        with torch.no_grad():
            cn = 0
            for batch_x, batch_y,  batch_x_mark, batch_y_mark in self.test_loader:
                batch_x = batch_x.float().to(self.args.device, non_blocking=pin_memory)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)

                if self.args.graph_input: 
                    pred = self.model(batch_x, self.args.sub_adj)
                else : 
                    if self.args.linear_output:
                        pred = self.model(batch_x)
                    else: 
                        pred = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            

                # print(data.y.shape, pred.shape)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = pred[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                loss += func.mse_loss(true, pred, reduction="mean")
                pred_.append(pred.data.numpy())
                truth_.append(true.data.numpy())
                cn += 1

                # print(cn)
                # if cn % 10 == 1: 
                #     visualize(data.cpu(), data.y.cpu(), pred.cpu())


            loss = loss/cn
            self.args.logger.info("[*] loss:{:.4f}".format(loss))
            pred_ = np.concatenate(pred_, 0)
            truth_ = np.concatenate(truth_, 0)
            mae = base_framework.metric(truth_, pred_, self.args)
            return loss
        
    @staticmethod
    def metric(ground_truth, prediction, args):
        global result
        pred_time = [args.y_len]
        args.logger.info("[*] phase {}, testing".format(args.phase))
        for i in pred_time:
            mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
            rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) 
            mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
            args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
            # print(result[i],i)            
            result[i]["mae"][args.phase] = mae
            result[i]["mape"][args.phase] = mape
            result[i]["rmse"][args.phase] = rmse
        return mae

    def run(self): 
        ##* init logger
        logger = init_log(self.args)  
        logger.info("params : %s", vars(self.args))

        #* multi-phase train 
        self.inc_state = False 
        for phase in range(self.args.begin_phase, self.args.end_phase+1):
            self.args.phase = phase 
            self.args.logger.info("[*] phase {} start training".format(self.args.phase)) 
            
            self.prepare()
            if self.args.train:
                self.train()
            else: 
                state_dict = torch.load(self.args.test_model_path, map_location=self.args.device)["model_state_dict"]
                self.model.load_state_dict(state_dict)

            self.test_model()
            self.inc_state = True 

        self.report_result()

def init_args(args): 
    #* complete args 
    with open("configs/" + args.conf + ".json", "r") as f:
        config = json.load(f)
        for key, value in config.items():
            if not hasattr(args, key):
                setattr(args, key, value)

    args.logname = args.conf
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    args.path = osp.join(args.exp_path, args.logname+args.time)
    args.pred_len = args.y_len
    ct.mkdirs(args.path)
    if args.train == False: args.load = False
    return args

def main(args):
    args = init_args(args) ##* initialize specialized args
    fm = base_framework(args)
    fm.run()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="incremental-naive")
    parser.add_argument("--data_name", type=str, default="electricity")
    parser.add_argument("--iteration", type=int, default=1)

    parser.add_argument("--auto_test", type=int, default=1)
    parser.add_argument("--load", action="store_true", default=False)
    parser.add_argument("--build_graph", action="store_true", default=False)
    parser.add_argument("--root_path", type=str, default="")
    parser.add_argument("--exp_path", type=str, default="exp/")
    parser.add_argument("--val_test_mix", action="store_true", default=False)
    parser.add_argument("--end_phase", type=int, default=1)
    parser.add_argument("--x_len", type=int, default=96)
    parser.add_argument("--y_len", type=int, default=96)
    args = parser.parse_args() 
    return args 

if __name__ == "__main__":
    args = parse_args() #* args needs adjust frequently and static
    seed_set(2021) 
    for i in range(args.iteration):
        main(args) #* run framework for one time 
