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
from torch.optim import lr_scheduler
from torch.optim import lr_scheduler

import random
from datetime import datetime


from models.TrafficStream import Basic_Model as TrafficStream 
from models import DLinear
from models import PatchTST 
from models import informer
from models import Source_Network, Target_Network
from models.TrafficStream import graph_constructor 
from utils import common_tools as ct
from utils.tools import visualize
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np
from models import detect
from models import replay
from models.ewc import EWC

from sklearn.preprocessing import StandardScaler
import pandas as pd

from data_process import get_dataset, data_provider
from torch.utils.data import DataLoader, dataloader, Dataset

result = {}
pin_memory = True 
num_workers = 4

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.lr if epoch < 3 else args.lr * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.lr}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.lr if epoch < 10 else args.lr*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.lr if epoch < 15 else args.lr*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.lr if epoch < 25 else args.lr*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.lr if epoch < 5 else args.lr*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))

def seed_set(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

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

class base_framework: 
    def __init__(self, args) -> None:
        self.args = args 

    def load_best_model(self):
        load_path = osp.join(self.args.exp_path, self.args.logname+self.args.time, str(self.args.phase-1), "best_model.pkl")
        state_dict = torch.load(load_path, map_location=self.args.device)["model_state_dict"]
        model = eval(self.args.model_name).Model(self.args).float()
        self.args.logger.info("[*] load from {}".format(load_path))
        model.load_state_dict(state_dict)
        model = model.to(self.args.device)
        self.model = model 

    def prepare(self): 

        self.inputs = get_dataset(self.args) 
        self.args.logger.info("[*] phase " + str(self.args.phase) + " Dataset load!")
        
        #* apply strategy 
        if self.inc_state:
            self.incremental_strategy()
        else:
            self.static_strategy()

        ##* prep dl 
        if self.args.train:
            self.score_loader = self.inputs["train_loader"]
            self.val_loader = self.inputs["val_loader"]
        self.test_loader = self.inputs["test_loader"]

    def incremental_strategy(self):  
        if self.args.load: 
            self.load_best_model() 
        load_path = osp.join(self.args.exp_path, self.args.logname+self.args.time, str(self.args.phase-1)) 
        # delete all the file end with .pkl
        # for file in os.listdir(load_path):
        #     if file.endswith(".pkl"):
        #         os.remove(os.path.join(load_path, file))
    
    def static_strategy(self):     
        self.S = Source_Network.Model(self.args).float().to(self.args.device)
        self.T = Target_Network.Model(self.args).float().to(self.args.device)
        global result
        result[self.args.pred_len] = {"mae":{}, "mape":{}, "rmse":{}}

    def pretrain_S(self): 
        _, self.train_loader = data_provider(args, 'train')     
                
        training_loss = 0.0 
        self.S.train() 
        cn = 0 
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark, label) in enumerate(self.train_loader):
            batch_x = batch_x.float().to(self.args.device)
            batch_y = batch_y.float().to(self.args.device)
            batch_x_mark = batch_x_mark.float().to(self.args.device)
            batch_y_mark = batch_y_mark.float().to(self.args.device)
            # print(np.sum(label.cpu().numpy()), np.sum(1-label.cpu().numpy()))

            self.optimizer_S.zero_grad()
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)

            if self.args.graph_input: 
                pred = self.S(batch_x, self.args.sub_adj)
            else : 
                if self.args.linear_output:
                    pred = self.S(batch_x)
                else: 
                    pred = self.S(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = pred[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

            pred = outputs
            true = batch_y
            # print("batch_x ", batch_y[1,1,1])
            # print("pred ", outputs, outputs.dtype)

            loss = self.lossfunc(pred, true, reduction="mean")
            # print(batch_x[1, 1, 1])

            # print("loss {:.7f}".format(loss.item()))
            # if self.args.ewc and self.inc_state:
            #     loss += self.model.compute_consolidation_loss()
            training_loss += float(loss)
            loss.backward()
            self.optimizer_S.step()
            
            cn += 1
            # if cn % 10 == 1: 
            #     visualize(data.cpu(), data.y.cpu(), pred.cpu())

            if self.args.lradj == 'TST':
                adjust_learning_rate(self.optimizer_S, self.scheduler_S, self.epoch + 1, self.args, printout=False)
                self.scheduler_S.step()
                
        training_loss = training_loss/cn 
        return training_loss

    def valid_S(self):
        self.S.eval()

        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark, label) in enumerate(self.val_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)

                if self.args.graph_input: 
                    pred = self.S(batch_x, self.args.sub_adj)
                else : 
                    if self.args.linear_output:
                        pred = self.S(batch_x)
                    else: 
                        pred = self.S(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = pred[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                loss = self.lossfunc(pred, true, reduction="mean")
                validation_loss += float(loss)
                # print(len(list(self.lossfunc(pred, true, reduction="none").mean(dim=0).detach().cpu().numpy())))
                # self.args.Score += self.lossfunc(pred, true, reduction="none").mean(dim=1).flatten().detach().cpu().tolist()
                cn += 1
        validation_loss = float(validation_loss/cn)
        self.validation_loss_list.append(validation_loss)
        return validation_loss
    
    def valid_T(self):
        self.T.eval()

        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark, label) in enumerate(self.val_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)

                if self.args.graph_input: 
                    pred = self.T(batch_x, self.args.sub_adj)
                else : 
                    if self.args.linear_output:
                        pred = self.T(batch_x)
                    else: 
                        pred = self.T(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = pred[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                loss = self.lossfunc(pred, true, reduction="mean")
                validation_loss += float(loss)
                # print(len(list(self.lossfunc(pred, true, reduction="none").mean(dim=0).detach().cpu().numpy())))
                # self.args.Score += self.lossfunc(pred, true, reduction="none").mean(dim=1).flatten().detach().cpu().tolist()
                cn += 1
        validation_loss = float(validation_loss/cn)
        # self.validation_loss_list.append(validation_loss)
        return validation_loss

    def get_Score(self):
        self.args.Score = []
        cn = 0
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark, label) in enumerate(self.score_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)

                if self.args.graph_input: 
                    pred = self.S(batch_x, self.args.sub_adj)
                else : 
                    if self.args.linear_output:
                        pred = self.S(batch_x)
                    else: 
                        pred = self.S(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = pred[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                # print(len(list(self.lossfunc(pred, true, reduction="none").mean(dim=0).detach().cpu().numpy())))
                self.args.Score += self.lossfunc(pred, true, reduction="none").mean(dim=1).flatten().detach().cpu().tolist()
                cn += 1

    def pretrain_T(self):
        _, self.train_loader = data_provider(args, 'train')     
            
        training_loss = 0.0 
        start_time = datetime.now() 

        self.T.train() 
        cn = 0 
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark, label) in enumerate(self.train_loader):
            batch_x = batch_x.float().to(self.args.device)
            batch_y = batch_y.float().to(self.args.device)
            batch_x_mark = batch_x_mark.float().to(self.args.device)
            batch_y_mark = batch_y_mark.float().to(self.args.device)
            # print(batch_x.shape)
            # print("batch_x ", batch_x[1,1,1])
            # print(batch_x[1,1,1])
            # print(self.model.state_dict()["enc_embedding.value_embedding.tokenConv.weight"])

            # count how many 1 and 0 in the label , print the number of 1 and 0
            # print(np.sum(label.cpu().numpy()), np.sum(1-label.cpu().numpy()))

            self.optimizer_T.zero_grad()
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)

            if self.args.graph_input: 
                pred = self.T(batch_x, self.args.sub_adj)
            else : 
                if self.args.linear_output:
                    pred = self.T(batch_x)
                else: 
                    pred = self.T(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = pred[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

            pred = outputs
            true = batch_y
            # print("batch_x ", batch_y[1,1,1])
            # print("pred ", outputs, outputs.dtype)

            loss = self.lossfunc(pred, true, reduction="none").mean(dim=1)
            loss = (loss * (1 - label.to(self.args.device))).mean()
            #! adjust here
            # print(batch_x[1, 1, 1])

            # print("loss {:.7f}".format(loss.item()))
            # if self.args.ewc and self.inc_state:
            #     loss += self.model.compute_consolidation_loss()
            training_loss += float(loss)
            loss.backward()
            self.optimizer_T.step()
            
            cn += 1
            # if cn % 10 == 1: 
            #     visualize(data.cpu(), data.y.cpu(), pred.cpu())

            if self.args.lradj == 'TST':
                adjust_learning_rate(self.optimizer_T, self.scheduler_T, self.epoch + 1, self.args, printout=False)
                self.scheduler_T.step()
        training_loss = training_loss/cn 
        return training_loss

    def joint_train(self):
        _, self.train_loader = data_provider(args, 'train')     
                
        training_loss = 0.0 
        self.S.train() 
        self.T.train() 
        cn = 0 
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark, label) in enumerate(self.train_loader):
            batch_x = batch_x.float().to(self.args.device)
            batch_y = batch_y.float().to(self.args.device)
            batch_x_mark = batch_x_mark.float().to(self.args.device)
            batch_y_mark = batch_y_mark.float().to(self.args.device)
            # print(batch_x.shape)
            # print("batch_x ", batch_x[1,1,1])
            # print(batch_x[1,1,1])
            # print(self.model.state_dict()["enc_embedding.value_embedding.tokenConv.weight"])

            # count how many 1 and 0 in the label , print the number of 1 and 0
            # print(np.sum(label.cpu().numpy()), np.sum(1-label.cpu().numpy()))

            self.optimizer_S.zero_grad()
            self.optimizer_T.zero_grad()
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)

            if self.args.graph_input: 
                pred_S = self.S(batch_x, self.args.sub_adj)
                pred_T = self.T(batch_x, self.args.sub_adj)
            else : 
                if self.args.linear_output:
                    pred_S, F_S = self.S(batch_x, feature=True)
                    pred_T, F_T = self.T(batch_x, feature=True)
                else: 
                    pred_S = self.S(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    pred_T = self.T(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        

            f_dim = -1 if self.args.features == 'MS' else 0
            pred_S = pred_S[:, -self.args.pred_len:, f_dim:]
            pred_T = pred_T[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

            true = batch_y
            # print("batch_x ", batch_y[1,1,1])
            # print("pred ", outputs, outputs.dtype)
            normal_mask = (1 - label).reshape(len(label),1,1,1).to(self.args.device)
            # print(F_S[2,0,2:3,23:29], F_T[2,0,2:3,23:29]) 
            loss_KD = self.lossfunc(pred_S * normal_mask, pred_T.detach() * normal_mask, reduction="mean")
            # [Batch, C，P, d ]
            loss_S = self.lossfunc(pred_S, true, reduction="mean")
            loss_T = self.lossfunc(pred_T, true, reduction="none").mean(dim=1)
            loss_T = (loss_T * (1 - label.to(self.args.device))).mean()
            loss = loss_S + loss_T + loss_KD
            # print(batch_x[1, 1, 1])

            # print("loss {:.7f}".format(loss.item()))
            # if self.args.ewc and self.inc_state:
            #     loss += self.model.compute_consolidation_loss()
            training_loss += float(loss)
            loss.backward()
            self.optimizer_S.step()
            self.optimizer_T.step()
            
            cn += 1
            # if cn % 10 == 1: 
            #     visualize(data.cpu(), data.y.cpu(), pred.cpu())

            if self.args.lradj == 'TST':
                adjust_learning_rate(self.optimizer_S, self.scheduler_S, self.epoch + 1, self.args, printout=False)
                adjust_learning_rate(self.optimizer_T, self.scheduler_T, self.epoch + 1, self.args, printout=False)
                self.scheduler_S.step()
                self.scheduler_T.step()

        training_loss = training_loss/cn 
        return training_loss

    def train(self): 
        # for i in self.model.parameters(): print(i)
        global result
        path = osp.join(self.args.path, str(self.args.phase))
        ct.mkdirs(path)

        ##* Model Optimizer
        self.optimizer_S = optim.Adam(self.S.parameters(), lr=self.args.lr)
        self.optimizer_T = optim.Adam(self.T.parameters(), lr=self.args.lr)
        if self.args.loss == "mse": self.lossfunc = func.mse_loss
        elif self.args.loss == "huber": self.lossfunc = func.smooth_l1_loss

        (self.args.pct_start,self.args.train_epochs,self.args.lr,train_steps) = (self.args.pct_start, self.args.epoch, self.args.lr, len(self.score_loader)-1)
        self.scheduler_T = lr_scheduler.OneCycleLR(optimizer = self.optimizer_T,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.lr)
        self.scheduler_S = lr_scheduler.OneCycleLR(optimizer = self.optimizer_S,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.lr)


        ##* train start
        self.args.logger.info("[*] phase " + str(self.args.phase) + " Training start")

        lowest_validation_loss = 1e7
        counter = 0
        patience = 100
        use_time = []
        self.validation_loss_list = []

        self.args.start_train = 1
        self.args.train_mode = 'pretrain'
        for self.epoch in range(self.args.epoch): #* train body 
            if self.args.train_mode == 'pretrain': 
                training_loss = self.pretrain_S()
                #todo train S()
                validation_loss = self.valid_S()
                self.get_Score()
                #todo val + score 
                self.args.logger.info(f"epoch:{self.epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")
                self.pretrain_T()
                ##
                validation_loss_T = self.valid_T()

                print("vs, vt", validation_loss, validation_loss_T)
                ##
                #todo train T()

                # Early Stop
                if validation_loss <= lowest_validation_loss:
                    counter = 0
                    lowest_validation_loss = validation_loss
                    save_model = self.S
                    if self.inc_state and self.args.ewc:
                        save_model = self.S.model 
                    torch.save({'model_state_dict': save_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+("_epoch_%d.pkl" % self.epoch)))
                else:
                    counter += 1  
                    if counter > patience:
                        break
                if self.args.lradj != 'TST':
                    adjust_learning_rate(self.optimizer_T, self.scheduler_T, self.epoch + 1, self.args)
                    adjust_learning_rate(self.optimizer_S, self.scheduler_S, self.epoch + 1, self.args)
                else:
                    print('Updating learning rate to {}'.format(self.scheduler_T.get_last_lr()[0]))
                    print('Updating learning rate to {}'.format(self.scheduler_S.get_last_lr()[0]))
                #todo post process() 
            elif self.args.train_mode == 'joint': 
                training_loss = self.joint_train()

                validation_loss = self.valid_S()
                ##
                validation_loss_T = self.valid_T()

                print("vs, vt", validation_loss, validation_loss_T)
                ##
                self.get_Score()

                self.args.logger.info(f"epoch:{self.epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")

                if validation_loss <= lowest_validation_loss:
                    counter = 0
                    lowest_validation_loss = validation_loss
                    save_model = self.S
                    if self.inc_state and self.args.ewc:
                        save_model = self.S.model 
                    torch.save({'model_state_dict': save_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+("_epoch_%d.pkl" % self.epoch)))
                else:
                    counter += 1  
                    if counter > patience:
                        break
                if self.args.lradj != 'TST':
                    adjust_learning_rate(self.optimizer_S, self.scheduler_S, self.epoch + 1, self.args)
                    adjust_learning_rate(self.optimizer_T, self.scheduler_T, self.epoch + 1, self.args)
                else:
                    print('Updating learning rate to {}'.format(self.scheduler_S.get_last_lr()[0]))
                    print('Updating learning rate to {}'.format(self.scheduler_T.get_last_lr()[0]))
            
            elif self.args.train_mode == 'normal':
                pass 

            self.args.train_mode = 'joint'


        # after training has been done
        
        epoch_idx = np.argmin(self.validation_loss_list)
        best_model_path = osp.join(path, str(round(lowest_validation_loss,4))+("_epoch_%d.pkl" % epoch_idx))
        
        best_S = Source_Network.Model(self.args).float()
        best_S.load_state_dict(torch.load(best_model_path, self.args.device)["model_state_dict"])
        torch.save({'model_state_dict': best_S.state_dict()}, osp.join(path, "best_model.pkl"))
        import os 
        if self.args.phase + 1 == self.args.end_phase and self.args.end_phase > 1:
            os.system('python main.py --conf ECL-DLinear_t --test_model_path {} > test.out'.format(osp.join(path, "best_model.pkl")))
        self.S = best_S
        self.S = self.S.to(self.args.device)        
        
    def report_result(self):
        global result
        for i in [self.args.pred_len]:
            for j in ['mae', 'rmse', 'mape']:
                info = ""
                for phase in range(self.args.begin_phase, self.args.end_phase):
                    if i in result:
                        if j in result[i]:
                            if phase in result[i][j]:
                                info+="{:.4f}\t".format(result[i][j][phase])
                self.args.logger.info("{}\t{}\t".format(i,j) + info)

    def test_model(self):
        self.S.eval()
        pred_ = []
        truth_ = []
        loss = 0.0
        with torch.no_grad():
            cn = 0
            for batch_x, batch_y,  batch_x_mark, batch_y_mark, label in self.test_loader:
                batch_x = batch_x.float().to(self.args.device, non_blocking=pin_memory)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)

                if self.args.graph_input: 
                    pred = self.S(batch_x, self.args.sub_adj)
                else : 
                    if self.args.linear_output:
                        pred = self.S(batch_x)
                    else: 
                        pred = self.S(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
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
        pred_time = [args.pred_len]
        args.logger.info("[*] phase {}, testing".format(args.phase))
        for i in pred_time:
            mae = masked_mae_np(ground_truth[:, -i:, :], prediction[:, -i:, :], 0)
            rmse = masked_mse_np(ground_truth[:, -i:, :], prediction[:, -i:, :], 0) 
            mape = masked_mape_np(ground_truth[:, -i:, :], prediction[:, -i:, :], 0)
            args.logger.info("T:{:d}\tMAE\t{:.6f}\tRMSE\t{:.6f}\tMAPE\t{:.6f}".format(i,mae,rmse,mape))
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
        for phase in range(self.args.begin_phase, self.args.end_phase):
            self.args.phase = phase 
            self.args.logger.info("[*] phase {} start training".format(self.args.phase)) 
            
            self.prepare()
            if self.args.train:
                self.train()
            else: 
                state_dict = torch.load(self.args.test_model_path, map_location=self.args.device)["model_state_dict"]
                self.S.load_state_dict(state_dict)

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
    args.num_workers = 4
    args.start_train = 0
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

    parser.add_argument("--load", action="store_true", default=True)
    parser.add_argument("--build_graph", action="store_true", default=False)
    parser.add_argument("--root_path", type=str, default="")
    parser.add_argument("--exp_path", type=str, default="exp/")
    parser.add_argument("--val_test_mix", action="store_true", default=False)
    # parser.add_argument("--end_phase", type=int, default=1)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--noise_rate", type=float)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--test_model_path", type=str, default="/Disk/fhyega/code/BASE/exp/ECL-PatchTST2023-07-23-21:59:42.618606/0/0.0379_epoch_25.pkl")
    parser.add_argument("--idx", type=int, default=213)
    args = parser.parse_args() 
    return args 

if __name__ == "__main__":
    args = parse_args() #* args needs adjust frequently and static
    seed_set(2021) 
    for i in range(args.iteration):
        main(args) #* run framework for one time 
