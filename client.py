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

parser = argparse.ArgumentParser(description='Client running')
parser.add_argument('--name','-n',help='client name',default='0')
parser.add_argument('--serverport','-p',help='server port',default='13597')
parser.add_argument('--serverip','-i',help='server ip',default='127.0.0.1')
parser.add_argument('--dataroot','-r',help='data root',default='data/traindata')
parser.add_argument('--datainfo',help='data information',default='data/info.json')
parser.add_argument('--cuda',help='cuda index',default='0')

class SameContrastiveLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x1):
        x1 = x1[:,0]
        return torch.sum(torch.abs(1-x1))

class DiffContrastiveLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x1):
        x1 = x1[:,0]
        return torch.sum(torch.abs(x1))
        
def train_core(net,loader,cuda,delay,cfg,epoch=None)->nn.Module:
    net = net.cuda(cuda)
    net = net.train()

    cross_loss = CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(net.parameters(),lr = cfg["lr"])
    
    if epoch==None:
        epoch = cfg['epoch']
    
    for epoch in range(epoch):            
        for batch_num,(data,label) in enumerate(loader):
            data = data.cuda(cuda)
            label = label.cuda(cuda)
                
            output = net(data)
            loss = cross_loss(output,label)
            # print(label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        time.sleep(delay)
        
    return net


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
def req_model_core(serverIp,serverPort):
    url = f'{url_build_core(serverIp,serverPort)}/req_model'
    data = {}
    while(True):
        try:
            r = requests.post(url,data=pickle.dumps(data))
            break
        except:
            pass
    return pickle.loads(r.content)

def req_cfg_core(serverIp,serverPort):
    url = f'{url_build_core(serverIp,serverPort)}/req_cfg'
    r = requests.get(url)
    return pickle.loads(r.content)

def info_build_core(datainfo):
    with open(datainfo) as f:
        info = json.loads(f.read())
    return info

def send_model_core(serverIp,serverPort,model,name,cuda):
    url = f'{url_build_core(serverIp,serverPort)}/send_model'
    data = {
        'name':name,
        'model':model,
        'cuda':cuda,
    }
    data = pickle.dumps(data)
    resp = requests.post(url,data)
    return resp.content

def delay_build_core(name,info):
    return info[name]['delay']


from data import CIFAR_10_Dataset
def loader_build_core(train_data_path,cfg):
    train_data = CIFAR_10_Dataset(train_data_path)
    train_data_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True,pin_memory=True)
    return train_data_loader

import glob
def label_list_build_core(data_path):
    label_path_list = glob.glob(f'{data_path}/*')
    label_list = [int(path.split('/')[-1]) for path in label_path_list]
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
        local_model = self.train()
        self.send_model(local_model)
    
    def info_build(self):
        return info_build_core(self.datainfo)
    
    def req_model(self):
        return req_model_core(self.ip,self.port)
    
    def req_cfg(self):
        return req_cfg_core(self.ip,self.port)
    
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
    
    def send_model(self,model):
        return send_model_core(self.ip,self.port,model,self.name,self.cuda)
    
        
    

if __name__ == "__main__":
    args = parser.parse_args()
    client = Client(args)
    client.run_train()

    