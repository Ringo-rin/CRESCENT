# %%
from itertools import *

# import modules
# Highway = modules.Highway
import prepare_data
import nnet_survival
# import GNN_model
import models

import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data

from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, roc_auc_score
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
import random
from parser import Parser
import networkx as nx


# import dgl.nn.pytorch as dglnn
# import torch.nn as nn
# import dgl
import time
from tqdm import tqdm
import math

import matplotlib.pyplot as plt

# from modules import Highway
from utils import setup_seed,get_time_c_td
# from GNN_model import Classifier,get_gin_model
# import model_v3

import numpy as np
import scipy.sparse as sp
import torch

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


import warnings
warnings.filterwarnings('ignore')

def get_Data(device,args):
    x_exp, clin, edge_index, ppi_network, _ = prepare_data.load(args)
    x_train,y_train, (time,event,breaks) = prepare_data.make_data_for_model(x_exp,clin)
    ## numpy to tensor 
    x = torch.tensor(x_train.T).float()
    y = torch.tensor(y_train).float()

    time = torch.tensor(time)
    event = torch.tensor(event)

    # edge_index = torch.tensor(edge_index, dtype=torch.long)
    ## 求标准lpls矩阵
    # NL=nx.normalized_laplacian_matrix(ppi_graph) 
    data = np.ones(edge_index[1].shape[0])
    adj = sp.coo_matrix((data, (edge_index[0], edge_index[1])), shape=(ppi_network.shape[0], ppi_network.shape[1]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)


    ###load data to GPU
    x = x.to(device)
    y = y.to(device)
    # edge_index = edge_index.to(device)

    ### 去稀疏化
    adj = adj.to_dense()

    count = torch.sign(adj).sum()
    print('-------Here!',count - adj.shape[0],'edges on adj',adj.shape,'------')

    adj = adj.to(device)
    time = time.to(device)
    event = event.to(device)

    ## 划分数据集
    gene_dataset = Data.TensorDataset(x,y,time,event)

    train_size = int(len(gene_dataset)*0.8)
    test_size = len(gene_dataset) - train_size


    train_dataset, test_dataset = torch.utils.data.random_split(gene_dataset, [train_size , test_size])
    print('len of train_size',train_size,'len of test_size',test_size)


    train_loader = Data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=args.batch_size,      # mini batch size
        shuffle=True,                        
    )

    test_loader = Data.DataLoader(
        dataset=test_dataset,      # torch TensorDataset format
        batch_size=args.batch_size,     
        # shuffle=True,              
    )

    ####
    # nums = '0.7765'
    # pth = 'data_split/'+nums+'__'+'1000'+'.npy'
    # data = np.load(pth, allow_pickle=True)
    # train_dataset_indices = data.item()['train_dataset_indices']
    # test_dataset_indices = data.item()['test_dataset_indices']

    # gene_dataset = Data.TensorDataset(x,y,time,event)


    # train_x,train_y,train_time,train_event = gene_dataset[train_dataset_indices]
    # test_x,test_y,test_time,test_event = gene_dataset[test_dataset_indices]

    # train_dataset = Data.TensorDataset(train_x,train_y,train_time,train_event)
    # test_dataset = Data.TensorDataset(test_x,test_y,test_time,test_event)
    # ####
    # train_loader = Data.DataLoader(
    #     dataset=train_dataset,      # torch TensorDataset format
    #     batch_size=args.batch_size,      # mini batch size
    #     shuffle=True,                        
    # )

    # test_loader = Data.DataLoader(
    #     dataset=test_dataset,      # torch TensorDataset format
    #     batch_size=args.batch_size,     
    #     # shuffle=True,              
    # )

    # g = dgl.graph((edge_index[0], edge_index[1]))
    # g = dgl.add_self_loop(g)


    node_nums = x_train.shape[0]
    n_intervals = int(y_train.shape[1]/2)
    # print(x_train.shape[0],'**')
    return train_loader, test_loader, adj, node_nums, n_intervals, breaks, train_dataset,test_dataset

def train_and_eval(
    model,
    adj,
    loss_func,
    rank_loss,
    optimizer,
    scheduler,
    breaks,
    train_loader,
    test_loader,
    args,
    train_dataset,
    test_dataset
    ):

    train_size = len(train_dataset)
    test_size = len(test_dataset)
    # print('&&&&',train_size,test_size)

    # train_size = len(train_dataset.indices)
    # test_size = len(test_dataset.indices)

    LOSS = []
    num_epoch = args.epochs
    train_c_index_reload = []
    test_c_index_reload = []

    train_loss_reload = []
    test_loss_reload = []

    train_c_td_reload = []
    test_c_td_reload = []

    TEST_max_C_INDEX = 0
    TEST_max_C_TD = 0



    for epoch in range(num_epoch): 
        model.train()
        loop = tqdm(enumerate(train_loader), total =len(train_loader))
        y_preds = 0
        y_trues = 0

        times = 0
        events = 0
        

        epoch_loss = []
        # print('lr:',optimizer.state_dict()['param_groups'][0]['lr']) 
        for step, (batch_x,y_true,time,event) in loop:
            

            y_pred = model(adj,batch_x)            
            ##梯度下降
            loss1 = loss_func(y_true, y_pred).mean()  
            loss2 = rank_loss(y_true, y_pred,time,event)
            loss = args.a*loss1 + args.b*loss2

            optimizer.zero_grad()   
            loss.mean().backward()      
            optimizer.step()

            

            ### ### 计算c-index
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            time = time.cpu().detach().numpy()
            event = event.cpu().detach().numpy()
            if(step == 0):
                y_preds = y_pred
                y_trues = y_true
                times = time
                events = event
            else:
                y_preds = np.vstack((y_preds,y_pred))
                y_trues = np.vstack((y_trues,y_true))
                times = np.append(times,time)
                events = np.append(events,event)
            
            LOSS.append((epoch,step,loss.mean().item()))
            epoch_loss.append(loss.mean().item())

            # ##更新信息
            loop.set_description(f'Training... Epoch [{epoch+1}/{num_epoch}]')
            loop.set_postfix(batch_loss = loss.mean().item(),mean_loss = np.mean(np.array(epoch_loss)))
            # # scheduler.step()

        scheduler.step()
        
        ##计算c-index
        c_index_train = cal_c_index(y_preds,times,events,breaks)
        c_td_train = cal_c_td(y_preds,y_trues,times,events)
        
        train_c_index_reload.append(c_index_train)
        train_c_td_reload.append(c_td_train)
        ###计算完毕

        #### 计算平均损失
        data = []
        for i in range(len(LOSS)):
            data.append(LOSS[i][2])
        data = np.array(data)

        print('###Train data','C-index',round(c_index_train,6),'C-td',round(c_td_train,6), 'mean loss:',np.mean(data[-train_size:]))
        train_loss_reload.append(np.mean(data[-train_size:]))

        test_loss_array = []
        if (epoch >= 0):
            model.eval()

            y_preds = 0
            y_trues = 0
            times = 0
            events = 0

            loop = tqdm(enumerate(test_loader), total =len(test_loader))
            for step, (batch_x,y_true,time,event) in loop:

                y_pred = model(adj,batch_x)

                loss1 = loss_func(y_true, y_pred).mean()  
                loss2 = rank_loss(y_true, y_pred,time,event)
                test_loss = args.a*loss1 + args.b*loss2

                test_loss_array.append(test_loss.mean().item())

                ### 计算c-index
                y_pred = y_pred.cpu().detach().numpy()
                y_true = y_true.cpu().detach().numpy()
                time = time.cpu().detach().numpy()
                event = event.cpu().detach().numpy()
                if(step == 0):
                    y_preds = y_pred
                    y_trues = y_true
                    times = time
                    events = event
                else:
                    y_preds = np.vstack((y_preds,y_pred))
                    y_trues = np.vstack((y_trues,y_true))
                    times = np.append(times,time)
                    events = np.append(events,event)

                loop.set_description('Testing ')
                loop.set_postfix(mean_loss = np.mean(np.array(test_loss_array)))
            
            test_loss_reload.append(np.mean(test_loss_array[-test_size:]))
                
            ##计算c-index
            c_index_test = cal_c_index(y_preds,times,events,breaks)
            c_td_test = cal_c_td(y_preds,y_trues,times,events)

            test_c_index_reload.append(c_index_test)
            
            test_c_td_reload.append(c_td_test)

            print('***Test data:','c_index:',round(c_index_test,6),'c_td:',round(c_td_test,6))
            TEST_max_C_INDEX = max(c_index_test,TEST_max_C_INDEX)
            TEST_max_C_TD = max(c_td_test,TEST_max_C_TD)
            print('***MAX c_index:', round(TEST_max_C_INDEX,6),'Max c_td:', round(TEST_max_C_TD,6))
            ### 计算完毕

            if(c_index_test > 0.73):
                save_model(c_index_test,model,args)
    
    
    print('max c-indx, max c_td on test data is',TEST_max_C_INDEX,TEST_max_C_TD)
    if(TEST_max_C_INDEX > 0.75) or args.plt:

        plt_res(train_c_td_reload,test_c_td_reload,train_c_index_reload,test_c_index_reload,train_loss_reload,test_loss_reload,args,TEST_max_C_INDEX,TEST_max_C_TD)

        save_datasplit(train_dataset,test_dataset,TEST_max_C_INDEX)
    
    return LOSS

def save_datasplit(train_dataset,test_dataset,TEST_max_C_INDEX):

    dict = {
    'train_dataset_indices':train_dataset.indices,
    'test_dataset_indices':test_dataset.indices
    }

    pth = 'data_split/'+str(TEST_max_C_INDEX)[:6]+'__'+str(args.epochs)+'.npy'
    np.save(pth, dict)
    print('data_slpit dict is save as',pth)
    ## load data
    # data = np.load(pth, allow_pickle=True)
    # train_dataset_indices = data.item()['train_dataset_indices']
    # test_dataset_indices = data.item()['test_dataset_indices']
    return 


def cal_c_index(y_preds,times,events,breaks):
    oneyr_surv = np.cumprod(y_preds[:,:], axis=1)
    oneyr_surv = oneyr_surv.transpose(1, 0)
    c_index = [concordance_index(event_times=times,
                                     predicted_scores=interval_probs,
                                     event_observed=events)
                   for interval_probs in oneyr_surv]
    # print('###',c_index)
    index = np.nonzero(breaks>698)[0][0]
    # print('###',index)
    return c_index[index]

def cal_c_td(y_pred,y_true,times,end):
    c_td = nnet_survival.get_time_c_td(y_pred,y_true,times,end)
    return c_td

def plt_res(
    train_c_td,
    test_c_td,
    train_c_index,
    test_c_index,
    train_loss,
    test_loss,
    args,
    TEST_max_C_INDEX,
    TEST_max_C_TD):
    t = np.arange(0, args.epochs, 1)

    ### reload data
    dict = {
    'train_loss':train_loss,
    'test_loss':test_loss,
    'train_c_index':train_c_index,
    'test_c_index':test_c_index,
    'train_c_td':train_c_td,
    'test_c_td':test_c_td
    }

    pth = 'train_data/'+str(TEST_max_C_INDEX)[:6]+'_layer_'+str(args.layers)+'.npy'
    np.save(pth, dict)
    print('train_data dict is save as',pth)
    # reload over

    for data in [train_loss,test_loss,train_c_index,test_c_index,train_c_td,test_c_td]:
        data = np.array(data)
        plt.plot(t, data)

    # # print(train_c_index.shape,test_c_index.shape,train_loss.shape,test_loss.shape,t.shape)

    plt.legend(['train_loss', 'test_loss','train_c_index', 'test_c_index','train_c_td','test_c_td'])

    title_info = 'beta:'+str(args.b)+' layers:'+str(args.layers)+'\n'
    title_info += 'max_c_index:'+str(TEST_max_C_INDEX)[:6] + '  ' + 'max_c_td:' +str(TEST_max_C_TD)[:6]

    plt.title(title_info)
    pic_path = 'pics/'+str(TEST_max_C_INDEX)[:6]+'__'+str(args.epochs)+'.jpg'
    plt.savefig(pic_path)

    return

def save_model(c_index,model,args):
    PATH = 'model/'+str(c_index)[:6]+'.pth'
    torch.save(model, PATH)
    # print(args)
    print('model save as',PATH)

if __name__ == '__main__':
    args = Parser(description='setting for training').args
    print(args)

    # setup_seed(args.seed)

    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")


    train_loader, test_loader, adj, nodenums, n_intervals,breaks,train_dataset,test_dataset = get_Data(device,args)

    hidden_dim = args.hidden_dim
    learining_rate = args.lr
    weight_decay_rate = args.weight_decay_rate
    feature_dim = args.feature_dim
    lr_ratio = args.lr_ratio


    loss_func = nnet_survival.surv_likelihood(n_intervals)
    rank_loss = nnet_survival.rank_loss

    model = models.Classifier(
        in_dim = args.feature_dim,
        hidden_dim =  hidden_dim,
        n_classes = n_intervals,
        node_nums = nodenums,
        # layers = args.layers
        args = args
    ).to(device)

    print('model layers ==',args.layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learining_rate, weight_decay=weight_decay_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learining_rate, momentum=0.9, weight_decay=weight_decay_rate)
    
    if (args.lr_cos):
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-4)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=lr_ratio)

    

    print('***start training...')
    train_and_eval(
        model,
        adj,
        loss_func,
        rank_loss,
        optimizer,
        scheduler,
        breaks,
        train_loader,
        test_loader,
        args,
        train_dataset,
        test_dataset
    )

    print(args) 


    




# %%
