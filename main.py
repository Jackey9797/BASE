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
from models.Enhancer import Enhancer
from models.TrafficStream import graph_constructor 
from utils import common_tools as ct
from utils.tools import visualize
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np
from models import replay
from models.ewc import EWC

from sklearn.preprocessing import StandardScaler
import pandas as pd

from data_process import get_dataset, data_provider
from torch.utils.data import DataLoader, dataloader, Dataset
import warnings
warnings.filterwarnings("ignore")

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
    def __init__(self, args):
        self.args = args 

    def prepare(self): 

        self.inputs = get_dataset(self.args) 
        self.args.logger.info("[*] phase " + str(self.args.phase) + " Dataset load!")
        
        #* apply strategy 
        if self.inc_state:
            pass
        else:
            self.static_strategy()

        ##* prep dl 
        if self.args.train:
            self.score_loader = self.inputs["train_loader"]
            self.val_loader = self.inputs["val_loader"]
        self.test_loader = self.inputs["test_loader"]

    def static_strategy(self):  #* init P and A    
        self.S = Source_Network.Model(self.args).float().to(self.args.device)
        if self.args.same_init:  
            import copy
            self.args.Base_T = copy.deepcopy(self.S.base_model)
            self.args.Base_T.configs = self.S.args
        self.T = Target_Network.Model(self.args).float().to(self.args.device)
        self.args.S = self.S 
        self.args.T = self.T 

        global result
        for i in [24, 36, 48, 60, 96, 192, 336, 720]:
            result[i] = {"mae":{}, "mape":{}, "rmse":{}}

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

            self.optimizer_S.zero_grad()
            if not self.args.linear_output:
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
            pred = pred[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

            loss = self.lossfunc(pred, batch_y, reduction="mean")
            # print(batch_x[1, 1, 1])

            training_loss += float(loss)
            loss.backward()
            self.optimizer_S.step()
            
            cn += 1

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
                if not self.args.linear_output:
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
                pred = pred[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

                pred = pred.detach().cpu()
                batch_y = batch_y.detach().cpu()
                
                loss = self.lossfunc(pred, batch_y, reduction="mean")
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
                if not self.args.linear_output:
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
                pred = pred[:, -self.args.pred_len:, f_dim:].detach().cpu()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                batch_y = batch_y.detach().cpu()

                loss = self.lossfunc(pred, batch_y, reduction="mean")
                validation_loss += float(loss)
                # print(len(list(self.lossfunc(pred, true, reduction="none").mean(dim=0).detach().cpu().numpy())))
                # self.args.Score += self.lossfunc(pred, true, reduction="none").mean(dim=1).flatten().detach().cpu().tolist()
                cn += 1
        validation_loss = float(validation_loss/cn)
        # self.validation_loss_list.append(validation_loss)
        if validation_loss < self.args.valid_loss_T : 
            self.args.valid_loss_T = validation_loss 
            import copy #todo
            # def get_weights_copy(model):
            #     weights_path = 'weights_temp.pt'
            #     torch.save(model.state_dict(), weights_path)
            #     return torch.load(weights_path)
            # self.T = get_weights_copy(self.T)
            self.args.best_T =  copy.deepcopy(self.T)
            self.args.best_Tloss = validation_loss
        return validation_loss

    def get_Score(self):
        # g = self.S 
        # self.S = self.T 
        self.S.eval()
        self.args.Score = []
        self.args.use_cm = False
        self.args.get_score = True
        cn = 0
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark, label) in enumerate(self.score_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)
                if not self.args.linear_output:
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
                pred = pred[:, -self.args.pred_len:, f_dim:].detach().cpu()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu()

                # print(len(list(self.lossfunc(pred, true, reduction="none").mean(dim=0).detach().cpu().numpy())))
                if len(self.args.Score) == 0 : self.args.Score = self.lossfunc(pred, batch_y, reduction="none").mean(dim=1).detach().cpu().numpy() if self.args.indie else self.lossfunc(pred, batch_y, reduction="none").mean(dim=1).mean(dim=1).detach().cpu().numpy()
                else : self.args.Score = np.concatenate([self.args.Score, self.lossfunc(pred, batch_y, reduction="none").mean(dim=1).detach().cpu().numpy() if self.args.indie else self.lossfunc(pred, batch_y, reduction="none").mean(dim=1).mean(dim=1).detach().cpu().numpy()])
                
                cn += 1 

        self.args.use_cm = True
        # self.S = g
        

    def pretrain_T(self):
        _, self.train_loader = data_provider(args, 'train')     
            
        training_loss = 0.0 

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
            if not self.args.linear_output:
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
            pred = pred[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

            true = batch_y
            # print("batch_x ", batch_y[1,1,1])
            # print("pred ", outputs, outputs.dtype)
            loss = self.lossfunc(pred, true, reduction="none").mean(dim=1)
            # print(label.shape, loss.shape)
            loss = (loss * (1 - label.to(self.args.device))).mean()
            if self.args.grad_norm: loss = loss * (len(label.flatten()) / label.sum()) 
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
        enhancer = Enhancer(args)
        cn = 0 

        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark, label) in enumerate(self.train_loader):
            batch_x = batch_x.float().to(self.args.device)
            batch_y = batch_y.float().to(self.args.device)
            batch_x_mark = batch_x_mark.float().to(self.args.device)
            batch_y_mark = batch_y_mark.float().to(self.args.device)
            
            self.optimizer_S.zero_grad()
            self.optimizer_T.zero_grad()
            if not self.args.linear_output:
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)

            self.args.best_T.train()
            if self.args.linear_output:
                pred_S, F_S = self.S(batch_x, feature=True)
                if self.args.refiner: 
                    _pred_S = self.S(batch_x, given_feature=F_S)
                pred_T = self.T(batch_x)
                with torch.no_grad():
                    _, F_T = self.args.best_T(batch_x, feature=True)
            else: 
                pred_S, F_S = self.S(batch_x, batch_x_mark, dec_inp, batch_y_mark, feature=True)
                if self.args.refiner: 
                    _pred_S = self.S(batch_x, batch_x_mark, dec_inp, batch_y_mark, given_feature=F_S)
                pred_T = self.T(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                with torch.no_grad():
                    _, F_T = self.args.best_T(batch_x, batch_x_mark, dec_inp, batch_y_mark, feature=True)

            f_dim = -1 if self.args.features == 'MS' else 0
            pred_S = pred_S[:, -self.args.pred_len:, f_dim:]
            if self.args.refiner: 
                _pred_S = _pred_S[:, -self.args.pred_len:, f_dim:]
            pred_T = pred_T[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

            true = batch_y

            loss_rec = 0
            loss_anchor = 0

            if self.args.enhance: #* train reconstructor
                normal_mask = (1 - label).reshape(len(label),label.shape[-1],1).to(self.args.device)
                anchor_F = F_T # B * C * D * P_num 
                anchor_F = anchor_F.reshape(-1, anchor_F.shape[-2], anchor_F.shape[-1]).permute(0, 2, 1)
                # print("lst", anchor_F.permute(0, 1, 3, 2).reshape(-1, anchor_F.shape[-2], anchor_F.shape[-1]).shape)
                # print(anchor_F.shape) # (B * C) * P_num * D 
                rec_F  = self.S.correction_module.Refiner.rec(anchor_F)
                # print(anchor_F.shape) # (B * C) * P_num * D 

                # print(rec_pred.shape, rec_F.shape, normal_mask.shape, anchor_F.shape) # check shape is right? 
                # print("w",anchor_F.shape, rec_F.shape)
                loss_rec = func.mse_loss(rec_F.reshape(normal_mask.shape[0], normal_mask.shape[1], -1) * normal_mask, anchor_F.reshape(normal_mask.shape[0], normal_mask.shape[1], -1) * normal_mask, reduction="mean") 
                #* see which way really work as AD scorer -> reconstruct F is OK
                loss_anchor += 2 * loss_rec


            loss_KD = 0   #* KD part
            if self.args.always_align: self.args.need_align = 1 
            if self.args.aligner and self.args.need_align: 
                normal_mask = (1 - label).reshape(len(label),label.shape[-1],1,1).to(self.args.device)
                # print(F_T.shape, F_S.shape, label.shape, normal_mask.shape)
                loss_KD = func.mse_loss(F_S * normal_mask, F_T.detach() * normal_mask, reduction="mean")
            # [Batch, C，P, d ]
             
            # print(self.args.rs_before, self.args.rs_after)
            loss_S = self.lossfunc(pred_S, true, reduction="mean") * self.args.omega 
            if self.args.refiner: loss_S  += self.lossfunc(_pred_S, true, reduction="mean") + self.args.rs_after * self.args.sup_weight # only influence ref 
            loss_T = self.lossfunc(pred_T, true, reduction="none").mean(dim=1)
            loss_T = (loss_T * (1 - label.to(self.args.device))).mean()
            if self.args.grad_norm: loss_T = loss_T * (len(label.flatten()) / label.sum()) 
            
            
            loss = loss_S + loss_T + loss_KD * self.args.alpha + loss_anchor * self.args.beta
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
                                            max_lr = self.args.lr) #! change here


        ##* train start
        self.args.logger.info("[*] phase " + str(self.args.phase) + " Training start")

        lowest_validation_loss = 1e7
        self.args.valid_loss_T = 1e7
        counter = 0
        patience = self.args.early_stop
        use_time = []
        self.validation_loss_list = []

        self.args.start_train = 1
        self.args.train_mode = 'pretrain'


        for self.epoch in range(self.args.epoch): #* train body 
            if self.args.train_mode == 'pretrain': 
                self.args.use_cm = False
                if not self.args.debugger == 2: 
                    training_loss = self.pretrain_S()
                    #todo train S()
                    validation_loss = self.valid_S()
                    self.get_Score()
                    #todo val + score 
                    self.args.logger.info(f"epoch:{self.epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")
                    

                    self.pretrain_T()
                    ##
                    validation_loss_T = self.valid_T()
                else : 
                    self.args.best_Tloss = 0
                    validation_loss, validation_loss_T = 1, 1
                    self.get_Score()
                    self.args.best_T = self.T

                self.args.use_cm = True

                print("vs, vt", validation_loss, validation_loss_T)
                if validation_loss > self.args.best_Tloss: 
                    self.args.need_align = True
                else :
                    self.args.need_align = False

                # Early Stop
                if validation_loss <= lowest_validation_loss:
                    counter = 0
                    lowest_validation_loss = validation_loss
                    save_model = self.S
                    torch.save({'model_state_dict': save_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+("_epoch_%d.pkl" % self.epoch)))
                    torch.save({'model_state_dict': self.T.state_dict()}, osp.join(path, ("T_acco.pkl")))
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
                
                #todo post process() 
            elif self.args.train_mode == 'joint': 
                training_loss = self.joint_train()

                validation_loss = self.valid_S()
                ##
                validation_loss_T = self.valid_T()

                print("vs, vt", validation_loss, validation_loss_T)
                if validation_loss > self.args.best_Tloss: 
                    self.args.need_align = True
                else :
                    self.args.need_align = False
                ##
                print("need align? -> ", self.args.need_align, self.args.best_Tloss)

                self.get_Score()

                self.args.logger.info(f"epoch:{self.epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")

                # Early Stop
                if validation_loss <= lowest_validation_loss:
                    counter = 0
                    lowest_validation_loss = validation_loss
                    save_model = self.S
                    torch.save({'model_state_dict': save_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+("_epoch_%d.pkl" % self.epoch)))
                    torch.save({'model_state_dict': self.T.state_dict()}, osp.join(path, ("T_acco.pkl")))
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
            
            self.args.train_mode = 'joint' #*
            
            # if self.epoch == 30: break
            if self.args.early_break: 
                if self.epoch == self.args.early_break: break
            #todo add early stop
            


        # after training has been done
        
        epoch_idx = np.argmin(self.validation_loss_list)
        best_model_path = osp.join(path, str(round(lowest_validation_loss,4))+("_epoch_%d.pkl" % epoch_idx))
        
        best_S = Source_Network.Model(self.args).float()
        best_S.load_state_dict(torch.load(best_model_path, self.args.device)["model_state_dict"])
        torch.save({'model_state_dict': best_S.state_dict()}, osp.join(path, "best_model.pkl"))
        torch.save({'model_state_dict': self.args.best_T.state_dict()}, osp.join(path, "best_T_model.pkl"))
        import os 
        if self.args.phase + 1 == self.args.end_phase and self.args.end_phase > 1:
            os.system('python main.py --conf ECL-DLinear_t --test_model_path {} > test.out'.format(osp.join(path, "best_model.pkl")))
        self.S = best_S
        self.S = self.S.to(self.args.device)        
        print("check", best_model_path, " & ", self.args.valid_loss_T)
        
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
        # self.T.eval()
        # self.S.base_model.model.head = self.T.base_model.model.head
        pred_ = []
        truth_ = []
        loss = 0.0
        a_sum = 0
        b_sum = 0
        with torch.no_grad():
            cn = 0
            for batch_x, batch_y,  batch_x_mark, batch_y_mark, label in self.test_loader:
                batch_x = batch_x.float().to(self.args.device, non_blocking=pin_memory)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)
                # pred = self.S(batch_x)
                # print(batch_x.shape, batch_y.shape)
                # if self.args.use_cm:
                # tmp = batch_x - self.args.rec.reshape(-1, batch_x.shape[2],batch_x.shape[1]).permute(0, 2, 1)
                # q1, q2 = torch.quantile(torch.abs(tmp), torch.tensor([0.25, 0.75],device=tmp.device), dim=-2, keepdim=True)
                # print(((q2 - q1) * 2  + q2).shape, (torch.abs(tmp) > ((q2 - q1) * 1.5 + q2)).shape)
                # batch_x[0, :, 2] -= tmp[0, :, 2] * (torch.abs(tmp) > ((q2 - q1) * 1.5 + q2))[0, :, 2]
                self.args.show = torch.ones_like(batch_x)

                if self.args.test_en: 
                    E = Enhancer(self.args)
                    if self.args.test_en == 1: batch_x = E.jitter(batch_x) 
                    if self.args.test_en == 2: batch_x = E.spike(batch_x)
                    if self.args.test_en == 3: batch_x = E.l_slope(batch_x)
                    if self.args.test_en == 4: batch_x = E.substitude(batch_x)
                    if self.args.test_en == 5: batch_x = E.set_zero(batch_x)
                    if self.args.test_en == 6: batch_x = E.point_jitter(batch_x)
                    if self.args.test_en == 7: batch_x = E.point_missing(batch_x)


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

                # self.args.use_cm = False 
                # _pred = self.S(batch_x)
                # outputs = _pred[:, -self.args.pred_len:, f_dim:]
                # _pred = outputs.detach().cpu()
                # self.args.use_cm = True 

                channel = 4

                # if self.args.train == 0 and self.args.debugger == 1 and self.args.train_mode != 'test': 
                #     print(cn + 2)
                #     import matplotlib.pyplot as plt 
                #     idx = 0 
                #     # plt.plot(batch_x[0,-336:,channel].cpu().detach().numpy(), color='g') 
                #     # show loss by text
                #     # plt.savefig('before{}.png'.format(cn))
                #     # plt.close()
                #     # show loss by text
                #     plt.plot(torch.cat([batch_x[idx,-336:,channel].cpu(), torch.zeros_like(batch_y[idx,:,channel]).cpu()]).detach().numpy(), color='g') 
                #     plt.plot(torch.cat([batch_x[idx,-336:,channel].cpu(), batch_y[idx,:,channel].cpu()]).detach().numpy(), color = 'orange')
                #     plt.plot(torch.cat([batch_x[idx,-336:,channel].cpu(), pred[idx,:,channel].cpu()]).detach().numpy(), color = "blue")
                #     plt.plot(torch.cat([batch_x[idx,-336:,channel].cpu(), _pred[idx,:,channel].cpu()]).detach().numpy(), color='grey', linestyle=':')
                #     plt.savefig('all{}.png'.format(cn))

                #     plt.close()

                #     import matplotlib.pyplot as plt 
                #     idx = 0 
                #     channel = 2
                #     plt.plot(batch_x[0,-336:,channel].cpu().detach().numpy(), color='g') 
                #     # plt.plot(self.args.rec.reshape(batch_x.shape[0], batch_x.shape[2],batch_x.shape[1]).permute(0, 2, 1)[0,-336:,channel].cpu().detach().numpy(), color='b') 
                #     # show loss by text
                #     plt.savefig('rec{}.png'.format(29))
                #     plt.close()
                loss += func.mse_loss(true, pred, reduction="mean")
                pred_.append(pred.data.numpy())
                truth_.append(true.data.numpy())
                cn += 1


                if self.args.train == 0 and self.args.debugger == 1 and self.args.train_mode != 'test': 

                    import matplotlib.pyplot as plt 
                    # tmp = batch_x[0,:,channel] - self.args.rec.reshape(batch_x.shape[0], batch_x.shape[2],batch_x.shape[1]).permute(0, 2, 1)[0,:,channel]
                    # q1, q2 = torch.quantile(torch.abs(tmp), torch.tensor([0.25, 0.75],device=tmp.device))
                    plt.plot(batch_x[0,-336:,channel].cpu().detach().numpy(), color='orange') 
                    # batch_x[0,:,2][torch.abs(tmp) > (q2 - q1) * 1.5 + q2] = self.args.rec.reshape(batch_x.shape[0], batch_x.shape[2],batch_x.shape[1]).permute(0, 2, 1)[0,:,2][torch.abs(tmp) > (q2 - q1) * 1.5 + q2]
                    # plt.plot(self.args.rec.reshape(batch_x.shape[0], batch_x.shape[2],batch_x.shape[1]).permute(0, 2, 1)[0,-336:,channel].cpu().detach().numpy(), color='b') 
                    plt.plot(batch_x[0,-336:,channel].cpu().detach().numpy(), color='g')
                    plt.plot(self.args.show.reshape(batch_x.shape[0], batch_x.shape[2],batch_x.shape[1]).permute(0, 2, 1)[0,-336:,channel].cpu().detach().numpy(), color='g')

                    plt.savefig(str(cn) + ".png")
                    plt.close()
                    # print(cn,"a",func.mse_loss(true[0,:,2], pred[0,:,2], reduction="mean"))
                    # a_sum += func.mse_loss(true[0,:,2], pred[0,:,2], reduction="mean").item()
                    # pred = self.S(batch_x)
                    # pred = pred.detach().cpu()
                    # print(cn,"b",func.mse_loss(true[0,:,2], pred[0,:,2], reduction="mean"))
                    # b_sum += func.mse_loss(true[0,:,2], pred[0,:,2], reduction="mean").item()
                    # print("compare", a_sum, b_sum)


                    # print(cn)
                    # if cn % 10 == 1: 
                    #     visualize(data.cpu(), data.y.cpu(), pred.cpu())

            self.args.train_mode = 'test'
            loss = loss/cn
            self.args.logger.info("[*] loss:{:.4f}".format(loss))
            pred_ = np.concatenate(pred_, 0)
            truth_ = np.concatenate(truth_, 0)
            mae, mse = base_framework.metric(truth_, pred_, self.args)
            return mae, mse
        
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
        return mae, rmse

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

                if self.args.mainrs: 
                    torch.save(self.S.state_dict(), osp.join('./mainresult/', str(self.args.model_name), str(self.args.data_name) + str(self.args.refiner) + str(self.args.pred_len) + "S.pkl"))
                    torch.save(self.args.best_T.state_dict(), osp.join('./mainresult/', str(self.args.model_name), str(self.args.data_name) + str(self.args.refiner) + str(self.args.pred_len) + "T.pkl"))
                if self.args.abl: 
                    torch.save(self.S.state_dict(), osp.join('./ablation/', str(self.args.model_name), str(self.args.data_name) + str(self.args.refiner) + str(self.args.aligner) + str(self.args.loss) + "S.pkl"))
                    torch.save(self.args.best_T.state_dict(), osp.join('./ablation/', str(self.args.model_name), str(self.args.data_name) + str(self.args.refiner) + str(self.args.aligner) + str(self.args.loss) + "T.pkl"))
            elif self.args.summary: 
                #* ds + model for all len and all noise type and have and not  
                df = [pd.DataFrame(columns=['len', 're', 'mse', 'mae']) for i in range(8)]
                
                for len_ in [96, 192, 336, 720]: 
                    self.args.pred_len = len_
                    self.test_loader = get_dataset(self.args)["test_loader"]
                    self.S = Source_Network.Model(self.args).float().to(self.args.device) 
                    o_path = osp.join('./mainresult/', str(self.args.model_name), str(self.args.data_name) + str(0) + str(len_)) 
                    self.S.load_state_dict(torch.load(o_path + "S.pkl", map_location=self.args.device)) 
                    self.args.refiner = 0 
                    for i in range(8): 
                        self.args.test_en = i 
                        mae, mse = self.test_model()
                        df[i] = df[i].append({'len': len_, 're': 0, 'mse': mse, 'mae': mae}, ignore_index=True)

                    re_path = osp.join('./mainresult/', str(self.args.model_name), str(self.args.data_name) + str(1) + str(len_)) 
                    self.S.load_state_dict(torch.load(re_path + "S.pkl", map_location=self.args.device)) 
                    self.args.refiner = 1
                    for i in range(8): 
                        self.args.test_en = i 
                        mae, mse = self.test_model()
                        df[i] = df[i].append({'len': len_, 're': 1, 'mse': mse, 'mae': mae}, ignore_index=True)
                    # self.T.load_state_dict(torch.load(o_path, map_location=self.args.device)) 
                
                # save df to './mainresult/'
                for i in range(8):
                    df[i].to_csv(osp.join('./mainresult/', str(self.args.model_name), str(self.args.data_name) + str(i) + "df.csv"))

            else: 
                state_dict = torch.load(self.args.test_model_path, map_location=self.args.device)["model_state_dict"]
                state_dict_T = torch.load(self.args.test_model_path[:-14] + "best_T_model.pkl", map_location=self.args.device)["model_state_dict"]
                #* wrong , need manual debug
                self.S.load_state_dict(state_dict, strict=False)
                self.T.load_state_dict(state_dict_T, strict=False)
                self.args.best_T = self.T 

                
                        

            self.test_model()
            self.inc_state = True 

        if self.args.train:
            self.report_result()
            if self.args.abl: 
                print("----*-----")               
                for i in range(8): 
                    self.args.test_en = i 
                    self.test_model()
                print("----*-----")

            self.args.use_cm = False 
            self.test_model()
            self.S.base_model.model.head = self.args.best_T.base_model.model.head
            self.args.use_cm = True 
            self.test_model()
            self.S = self.args.best_T.to(self.args.device) 
            self.test_model()
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
    args.train_mode = 'pretrain'
    args.get_score = False
    args.use_cm = True

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
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--conf", type=str, default="incremental-naive")
    parser.add_argument("--data_name", type=str, default="electricity")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--mainrs", type=int, default=0)
    parser.add_argument("--abl", type=int, default=0)
    parser.add_argument("--abl_tmp_context", type=int, default=0)
    parser.add_argument("--abl_ae", type=int, default=0)

    parser.add_argument("--load", action="store_true", default=True)
    parser.add_argument("--build_graph", action="store_true", default=False)
    parser.add_argument("--same_init", action="store_true", default=True)
    parser.add_argument("--grad_norm", action="store_true", default=False)
    parser.add_argument("--refiner_residual", type=int, default=0)
    parser.add_argument("--root_path", type=str, default="")
    parser.add_argument("--exp_path", type=str, default="exp/")
    parser.add_argument("--val_test_mix", action="store_true", default=False)
    # parser.add_argument("--end_phase", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lradj", type=str, default="TST")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fc_dropout", type=float, default=0.2)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--noise_rate", type=float)
    parser.add_argument("--device", type=str, default="cuda:0")
    # parser.add_argument("--test_model_path", type=str, default="/Disk/fhyega/code/BASE/exp/ECL-PatchTST2023-08-19-16:00:30.039043/0/best_model.pkl")
    parser.add_argument("--test_model_path", type=str, default="/Disk/fhyega/code/BASE/exp/ECL-PatchTST2023-08-26-13:08:31.837686/0/best_model.pkl")
    parser.add_argument("--idx", type=int, default=213)
    parser.add_argument("--aligner", type=int, default=0)
    parser.add_argument("--always_align", type=int, default=1)
    parser.add_argument("--refiner", type=int, default=0)
    parser.add_argument("--rec_block_num", type=int, default=1)
    parser.add_argument("--enhance", type=int, default=0)
    parser.add_argument("--enhance_type", type=int, default=5)
    parser.add_argument("--seed", type=int, default=34)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--share_head", type=int, default=0)
    parser.add_argument("--add_noise", type=int, default=1)
    parser.add_argument("--add_norm", type=int, default=0)

    parser.add_argument("--jitter_sigma", type=float, default=0.4)
    parser.add_argument("--slope_rate", type=float, default=0.01)
    parser.add_argument("--slope_range", type=float, default=0.2)

    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--feature_jittering", type=int, default=1)
    parser.add_argument("--rec_intra_feature", type=int, default=0)
    parser.add_argument("--rec_ori", type=int, default=1)
    parser.add_argument("--mid_dim", type=int, default=128)
    parser.add_argument("--test_en", type=int, default=0)
    parser.add_argument("--debugger", type=int, default=0)
    parser.add_argument("--indie", type=int, default=1)
    parser.add_argument("--summary", type=int, default=0)

    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=1.1)
    parser.add_argument("--mask_border", type=int, default=1)
    parser.add_argument("--sup_weight", type=float, default=10.0)
    parser.add_argument("--rec_length_ratio", type=float, default=0.8)
    parser.add_argument("--ref_dropout", type=float, default=0.0)
    parser.add_argument("--ref_block_num", type=int, default=2)
    parser.add_argument("--add_FFN", type=int, default=0)
    parser.add_argument("--add_residual", type=int, default=0)
    parser.add_argument("--rec_all", type=int, default=0)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--early_break", type=int, default=0)
    parser.add_argument("--early_stop", type=int, default=10)
    

    args = parser.parse_args() 
    return args 

if __name__ == "__main__":
    args = parse_args() #* args needs adjust frequently and static
    seed_set(args.seed) 
    for i in range(args.iteration):
        main(args) #* run framework for one time 
