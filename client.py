import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import json
import os
import time
import pickle
import requests
import numpy as np
from client_global_variable import Client_Status
import traceback
import logging
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Client running')
parser.add_argument('--name','-n',help='client name',default='0')
parser.add_argument('--serverport','-p',help='server port',default='13597')
parser.add_argument('--serverip','-i',help='server ip',default='127.0.0.1')
parser.add_argument('--dataroot','-r',help='data root',default='data/traindata')
parser.add_argument('--datainfo',help='data information',default='data/info.json')
parser.add_argument('--cuda',help='cuda index',default='0')



def compute_center(features):
    # 计算特征的中心
    return torch.mean(features, dim=0)

def mse_loss(feature,c):
    
    feature =  F.normalize(feature, p=2, dim=1)
    feature = feature.mean(dim=0)
    # print(feature.shape)
    # print(c.shape)
    return F.mse_loss(feature,c)

def msc_loss(x_i, x_j, tau=0.1, epsilon=1e-8):
    # 提取特征并归一化
    z_i = F.normalize(x_i, p=2, dim=1)
    z_j = F.normalize(x_j, p=2, dim=1)
    
    # 计算特征中心
    c = compute_center(z_i)
    
    # 防止 c 出现 NaN 或者无穷大
    c = F.normalize(c, p=2, dim=0)
    
    # 减去中心并再次归一化
    z_i = z_i - c
    z_j = z_j - c
    
    z_i = F.normalize(z_i, p=2, dim=1)
    z_j = F.normalize(z_j, p=2, dim=1)
    
    # 计算相似性
    sim_ij = F.cosine_similarity(z_i, z_j)
    sim_ij = torch.exp(sim_ij / tau)
    
    # 计算分母
    sim_matrix = torch.exp(F.cosine_similarity(z_i.unsqueeze(1), z_i.unsqueeze(0), dim=2) / tau)
    sim_matrix = sim_matrix.sum(dim=1) - sim_ij + epsilon  # 添加 epsilon 以防止除以零
    
    # 计算损失
    loss = -torch.log(sim_ij / sim_matrix)
    
    return loss.mean(), c



class SameContrastiveLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x,label):
        score = (1-label)*x-label*x
        if torch.nan in score:
            print(x)
        return torch.sum(score)

class DiffContrastiveLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x1):
        x1 = x1[:,0]
        return torch.sum(torch.abs(x1))
    
    
        
def train_core(net,loaders,cuda,delay,cfg,epoch=None)->nn.Module:
    net = net.cuda(cuda)
    net = net.train()
    loader_0 = loaders[0]
    loader_1 = loaders[1]
    
    optimizer = torch.optim.SGD(net.parameters(),lr = cfg["lr"])
    
    if epoch==None:
        epoch = cfg['epoch']
    
    center =torch.as_tensor(cfg["center"]).cuda(cuda)
    c_sum = None
    count = 0
    
    for epoch in range(epoch):            
        for batch_num,(data_1,data_2) in enumerate(loader_0):
            data_1 = data_1.cuda(cuda)
            data_2 = data_2.cuda(cuda)
                
            output_1 = net(data_1)
            output_2 = net(data_2)
            # print(output_1)
            loss,c = msc_loss(output_1,output_2)
            # print(loss.item())
            if c_sum==None:
                c_sum = c.detach()
            else:
                c_sum+=c.detach()
            count+=1
            # print(label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    for epoch in range(3):
        for batch_num,(data_1,label_1,data_2,label_2) in enumerate(loader_1):
            
            data_1 = data_1.cuda(cuda)
            label_1 = label_1.cuda(cuda)
            output_1 = net(data_1,train=True)
            loss_cross = F.cross_entropy(output_1,label_1)
            # print(loss_cross.item())
            optimizer.zero_grad()
            loss_cross.backward()
            optimizer.step()
            
            data_2 = data_2.cuda(cuda)
            label_2 = label_2.cuda(cuda)
            output_2 = net(data_2,train=True)
            loss_cross = F.cross_entropy(output_2,label_2)
            # print(loss_cross.item())
            optimizer.zero_grad()
            loss_cross.backward()
            optimizer.step()
    
    c_sum/=count
    c_sum = c_sum.cpu()
    return net,c_sum


def compute_accuracy(possibility, label):
    sample_num = label.size(0)
    _, index = torch.max(possibility, 1)
    correct_num = torch.sum(label == index)
    return (correct_num/sample_num).item()

def test(net,loader,name):
    net.eval()
    acc = 0
    num = 0
    for batch_num,(image,label) in enumerate(loader):
        num += 1
        image = image.cuda(Client_Status.CUDA)/255.0
        label = label.cuda(Client_Status.CUDA)
        output = net(image)
        acc+= compute_accuracy(output,label)
            
    acc/= num
    if not(name in Client_Status.MAX_ACC.keys()):
        Client_Status.MAX_ACC[name] = acc
    elif acc>Client_Status.MAX_ACC[name]:
        Client_Status.MAX_ACC[name] = acc
        
    logging.info("{:s} max accuracy: {:.2f} current accuracy: {:.2f}"
              .format(name, Client_Status.MAX_ACC[name],acc*100))



def url_build_core(serverIp,serverPort):
    return f'http://{serverIp}:{serverPort}/server'

import random
def req_model_core(serverIp,serverPort,label):
    url = f'{url_build_core(serverIp,serverPort)}/req_model'
    data = {
        "label":label
    }
    while(True):
        try:
            r = requests.post(url,data=pickle.dumps(data))
            break
        except:
            pass
    return pickle.loads(r.content)

def req_cfg_core(serverIp,serverPort,label_list):
    url = f'{url_build_core(serverIp,serverPort)}/req_cfg'
    data = {
        "label_list":label_list
    }
    data = pickle.dumps(data)
    r = requests.post(url,data=data)
    return pickle.loads(r.content)

def info_build_core(datainfo):
    with open(datainfo) as f:
        info = json.loads(f.read())
    return info

def send_model_core(serverIp,serverPort,model,c,name,cuda,label):
    url = f'{url_build_core(serverIp,serverPort)}/send_model'
    data = {
        'name':name,
        'model':model,
        'cuda':cuda,
        'label':label,
        'feature':c
    }
    data = pickle.dumps(data)
    resp = requests.post(url,data)
    return resp.content

def delay_build_core(name,info):
    return info[name]['delay']


from data import Class_Wise_Dataset,Class_Wise_Con_Dataset
def loader_build_core(train_data_path,cfg):
    train_data_0 = Class_Wise_Dataset(train_data_path,cfg["label"])
    train_data_loader_0 = DataLoader(train_data_0, batch_size=cfg['batch_size'], shuffle=True,pin_memory=True)
    train_data_1 = Class_Wise_Con_Dataset(train_data_path,cfg["label"])
    train_data_loader_1 = DataLoader(train_data_1, batch_size=cfg['batch_size'], shuffle=True,pin_memory=True)
    return [train_data_loader_0,train_data_loader_1]

import glob
def label_list_build_core(data_path):
    label_list = []
    label_path_list = glob.glob(f'{data_path}/*')
    total_num = len(glob.glob(f'{data_path}/*/*.png'))
    for label in label_path_list:
        label_num = len(glob.glob(f"{label}/*.png"))
        if label_num/total_num>0.2:
            label_list.append(int(label.split('/')[-1]))            
    return label_list


class Client():
    def __init__(self, args) -> None:
        self.init_pass_(args)
    
    def init_pass_(self, args):
        
        self.name = args.name
        self.port = args.serverport
        self.ip = args.serverip
        self.dataroot = args.dataroot
        self.datainfo = args.datainfo
        self.cuda = int(args.cuda)
        
        self.info = self.info_build()
        self.delay = self.delay_build()
        self.label_list = self.label_list_build()
        
        self.cfg = self.req_cfg()
        self.loader = self.loader_build()
        self.model = self.req_model()
        
    
    def run_train(self):
        local_model,c = self.train()
        self.send_model(local_model,c,self.cfg["label"])
    
    def info_build(self):
        return info_build_core(self.datainfo)
    
    def req_model(self):
        return req_model_core(self.ip,self.port,self.cfg["label"])
    
    def req_cfg(self):
        return req_cfg_core(self.ip,self.port,self.label_list)
    
    def loader_build(self):
        data_path = f"{self.dataroot}/{self.name}"
        return loader_build_core(data_path,self.cfg)
    
    def label_list_build(self):
        data_path = f"{self.dataroot}/{self.name}"
        return label_list_build_core(data_path)
    
    def delay_build(self):
        return delay_build_core(self.name,self.info)
    
    def train(self):
        return train_core(self.model,self.loader,self.cuda,self.delay,self.cfg)
    
    def send_model(self,model,c,label):
        return send_model_core(self.ip,self.port,model,c,self.name,self.cuda,label)
    
        
    

if __name__ == "__main__":
    args = parser.parse_args()
    client = Client(args)
    client.run_train()

    