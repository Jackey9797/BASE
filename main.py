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
from utils import common_tools as ct
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np
from models import detect
from models import replay
from models.ewc import EWC


from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import to_dense_batch, k_hop_subgraph
from torch_geometric.data import Data, Dataset
import networkx as nx


result = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}}}
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



def z_score(data):
    return (data - np.mean(data)) / np.std(data)

class PEMS3_Stream: #* 从PEMS raw data中生成训练验证测试数据集 + graph
    def __init__(self, args, savepath, train_rate=0.6, val_rate=0.2, test_rate=0.2, val_test_mix=False):
        raw_data = np.load(osp.join(args.raw_data_path, str(args.year)+".npz"))["x"]
        
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
        graph = nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"])
        args.graph_size = graph.number_of_nodes() #! wait to be used 
        edge_index = np.array(list(graph.edges)).T   #* PEMS data 包含2部分，一部分是graph，一部分是sensor data 
        del graph

        np.savez(savepath, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, edge_index=edge_index)
        # prepared_data = {"train_x":train_x, "train_y":train_y, "val_x":val_x, "val_y":val_y, "test_x":test_x, "test_y":test_y, "edge_index":edge_index}
        # return prepared_data
    
    def slice_dataset(self, idx, x_len=12, y_len=12):
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
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T) #* exchange the dimension of Time and Node 
        y = torch.Tensor(self.y[index].T)
        return Data(x=x, y=y)  #* the input form is a graph, x and y seen as attribute of each node


def get_dataset(args):
    if args.data_name == 'PEMS3-Stream': 
        if args.data_process:
            PEMS3_Stream(args, osp.join(args.save_data_path, str(args.year)+'_30day'), val_test_mix=True)
        inputs = np.load(osp.join(args.save_data_path, str(args.year)+"_30day.npz"), allow_pickle=True)
    
    return inputs 

def init_args(args): 
    static_args = {
        ##* static args
        "load_config": "configs/", 
        "data_process": False ,
        "auto_test": 1,
        "load": True,
        "device": "cuda:0",
        
        ##* dataset related args
        "data_name": "PEMS3-Stream",
        "raw_data_path": "data/district3F11T17/finaldata/",
        "graph_path": "data/district3F11T17/graph/",
        "save_data_path": "data/district3F11T17/FastData/",
        "model_path": "exp/district3F11T17/",
        "year": 2012,
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

    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    args.path = osp.join(args.model_path, args.logname+args.time)
    ct.mkdirs(args.path)
    return args

class base_framework: 
    def __init__(self, args) -> None:
        self.args = args 

    def load_best_model(self):
        load_path = osp.join(self.args.model_path, self.args.logname+self.args.time, str(self.args.year-1), "best_model.pkl")
        ##!! file name file name file name 
    
        self.args.logger.info("[*] load from {}".format(load_path))
        state_dict = torch.load(load_path, map_location=self.args.device)["model_state_dict"]
        if 'tcn2.weight' in state_dict:
            del state_dict['tcn2.weight']
            del state_dict['tcn2.bias']

        #* model + load_state_dict
        model = TrafficStream(self.args) #TODO   modify model name
        model.load_state_dict(state_dict)
        model = model.to(self.args.device)
        self.model = model 

    def prepare(self): 
        ##* 1.init graph 
        init_graph(self.args)               
        ##* 2.prep complete data (form as follow ) 
        self.inputs = get_dataset(self.args) 
        # to show: print(inputs["train_x"].shape, inputs["train_y"].shape, inputs["val_x"].shape, inputs["val_y"].shape, inputs["test_x"].shape, inputs["test_y"].shape, inputs["edge_index"].shape)
        self.args.logger.info("[*] Year " + str(self.args.year) + " Dataset load!")
         
        ##* 3.apply certain strategy (like get select nodes and construct subgraph)
        if self.inc_state:
            self.incremental_strategy()
        else:
            self.static_strategy()

        ##* prep dl 
        self.train_loader = DataLoader(TrafficDataset(self.args.subgraph.numpy(), self.inputs, "train"), batch_size=self.args.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=n_work)
        self.val_loader = DataLoader(TrafficDataset(self.args.subgraph.numpy(), self.inputs,"val"), batch_size=self.args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
        self.test_loader = DataLoader(TrafficDataset(np.arange(self.args.graph_size), self.inputs,"test"), batch_size=self.args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
       
        

    def incremental_strategy(self):  #! incremental prepare work 目的是获得增量节点及其对应的subgraph
        
        if self.args.subgraph_train:
            node_list = list()
            ##* increase nodes
            if self.args.increase:
                cur_node_size = np.load(osp.join(self.args.graph_path, str(self.args.year)+"_adj.npz"))["x"].shape[0]
                pre_node_size = np.load(osp.join(self.args.graph_path, str(self.args.year-1)+"_adj.npz"))["x"].shape[0] #! args.year changed?
                node_list.extend(list(range(pre_node_size, cur_node_size)))
                pass

            ##* influence nodes
            if self.args.detect: 
                self.args.logger.info("[*] detect strategy {}".format(self.args.detect_strategy))
                pre_data = np.load(osp.join(self.args.raw_data_path, str(self.args.year-1)+".npz"))["x"]
                cur_data = np.load(osp.join(self.args.raw_data_path, str(self.args.year)+".npz"))["x"]
                pre_graph = np.array(list(nx.from_numpy_matrix(np.load(osp.join(self.args.graph_path, str(self.args.year-1)+"_adj.npz"))["x"]).edges)).T
                cur_graph = np.array(list(nx.from_numpy_matrix(np.load(osp.join(self.args.graph_path, str(self.args.year)+"_adj.npz"))["x"]).edges)).T
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
            cur_graph = torch.LongTensor(np.array(list(nx.from_numpy_matrix(np.load(osp.join(self.args.graph_path, str(self.args.year)+"_adj.npz"))["x"]).edges)).T)
            edge_list = list(nx.from_numpy_matrix(np.load(osp.join(self.args.graph_path, str(self.args.year)+"_adj.npz"))["x"]).edges)
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

            self.args.logger.info("number of increase nodes:{}, nodes after {} hop:{}, total nodes this year {}".format\
                        (len(node_list), self.args.num_hops, self.args.subgraph.size(), self.args.graph_size))
            self.args.node_list = np.asarray(node_list)

            graph = nx.Graph()
            graph.add_nodes_from(range(self.args.subgraph.size(0)))
            graph.add_edges_from(self.args.subgraph_edge_index.numpy().T)
            adj = nx.to_numpy_array(graph)
            adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
            self.args.sub_adj = torch.from_numpy(adj).to(torch.float).to(self.args.device) 
        else : 
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
        self.args.subgraph = torch.arange(self.args.graph_size) 
        self.args.sub_adj = self.args.adj   

        if self.args.year == self.args.begin_year: 
            self.model = TrafficStream(self.args).to(self.args.device)
        if self.args.strategy == "static" and self.args.year > self.args.begin_year: 
            self.args.train = False 
        if self.args.strategy == "retrain": 
            self.model = TrafficStream(self.args).to(self.args.device)


    def train(self): 
        global result
        path = osp.join(self.args.path, str(self.args.year))
        ct.mkdirs(path)

        ##* Model Optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        if self.args.loss == "mse": lossfunc = func.mse_loss
        elif self.args.loss == "huber": lossfunc = func.smooth_l1_loss


        ##* train start
        self.args.logger.info("[*] Year " + str(self.args.year) + " Training start")
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
            for batch_idx, data in enumerate(self.train_loader):
                if epoch == 0 and batch_idx == 0:
                    self.args.logger.info("node number {}".format(data.x.shape))
                data = data.to(self.args.device, non_blocking=pin_memory)
                
                optimizer.zero_grad()
                pred = self.model(data, self.args.sub_adj)
            
                if self.args.subgraph_train and self.inc_state:
                    pred, _ = to_dense_batch(pred, batch=data.batch)
                    data.y, _ = to_dense_batch(data.y, batch=data.batch) ##? 不知所云
                    pred = pred[:, self.args.mapping, :]
                    data.y = data.y[:, self.args.mapping, :] 
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
                for batch_idx, data in enumerate(self.val_loader):
                    data = data.to(self.args.device, non_blocking=pin_memory)
                    pred = self.model(data, self.args.sub_adj)
                    if self.args.subgraph_train and self.inc_state:
                        pred, _ = to_dense_batch(pred, batch=data.batch)
                        data.y, _ = to_dense_batch(data.y, batch=data.batch)
                        pred = pred[:, self.args.mapping, :]
                        data.y = data.y[:, self.args.mapping, :]
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
        best_model = TrafficStream(self.args)
        best_model.load_state_dict(torch.load(best_model_path, self.args.device)["model_state_dict"])
        torch.save({'model_state_dict': best_model.state_dict()}, osp.join(path, "best_model.pkl"))
        self.model = best_model
        self.model = self.model.to(self.args.device)        
        
        result[self.args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
        self.args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))
    
    def report_result(self):
        global result
        for i in [3, 6, 12]:
            for j in ['mae', 'rmse', 'mape']:
                info = ""
                for year in range(self.args.begin_year, self.args.end_year+1):
                    if i in result:
                        if j in result[i]:
                            if year in result[i][j]:
                                info+="{:.2f}\t".format(result[i][j][year])
                self.args.logger.info("{}\t{}\t".format(i,j) + info)

        for year in range(self.args.begin_year, self.args.end_year+1):
            if year in result:
                info = "year\t{}\ttotal_time\t{}\taverage_time\t{}\tepoch\t{}".format(year, result[year]["total_time"], result[year]["average_time"], result[year]['epoch_num'])
                self.args.logger.info(info)

    def test_model(self):
        self.model.eval()
        pred_ = []
        truth_ = []
        loss = 0.0
        with torch.no_grad():
            cn = 0
            for data in self.test_loader:
                data = data.to(self.args.device, non_blocking=pin_memory)
                pred = self.model(data, self.args.adj)
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
        pred_time = [3,6,12]
        args.logger.info("[*] year {}, testing".format(args.year))
        for i in pred_time:
            mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
            rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
            mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
            args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
            result[i]["mae"][args.year] = mae
            result[i]["mape"][args.year] = mape
            result[i]["rmse"][args.year] = rmse
        return mae

    def run(self): 
        ##* logger、dir prep
        logger = init_log(self.args)  
        logger.info("params : %s", vars(self.args))
        ct.mkdirs(self.args.save_data_path)

        ##* multi-phase train 
        self.inc_state = False 
        for year in range(self.args.begin_year, self.args.end_year+1): ##todo 将第一年拿出来单独处理 
            self.args.year = year 

            self.args.logger.info("[*] Year {} load from {}_30day.npz".format(self.args.year, osp.join(self.args.save_data_path, str(year)))) 
            
            self.prepare()
            if self.args.train:
                self.train()

            self.test_model()

            if self.args.strategy == "incremental": self.inc_state = True 

        self.report_result()

def init_graph(args):
    adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
    adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
    args.adj = torch.from_numpy(adj).to(torch.float).to(args.device) #* adj -> normalized 邻接矩阵
    args.graph_size = adj.shape[0]

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
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # args = None  ##* pre set args
    seed_set(13) ##* set seed
    main(args) ##* run any framework for a time 
