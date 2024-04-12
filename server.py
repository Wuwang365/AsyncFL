import threading
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from model import ResNet as Model
from flask import Blueprint, request
import json
import time
import pickle
from server_global_variable import Server_Status
import logging
import copy

server_app = Blueprint("server", __name__, url_prefix="/server")

server_status = Server_Status()


@server_app.route("/req_cfg", methods=["GET"])
def req_cfg():
    config = {
        "epoch": 2,
        "lr": 1e-3,
        "batch_size": 256
    }
    return pickle.dumps(config)


@server_app.route("/send_model", methods=["POST"])
def send_model():
    client_info = pickle.loads(request.data)
    net = client_info["model"].cpu()
    name = client_info['name']
    cuda = client_info['cuda']
    server_status._instance_lock.acquire()
    send_model_core(net,name,cuda,server_status)
    server_status._instance_lock.release()
    return pickle.dumps("")

@server_app.route("/req_model", methods=["POST"])
def req_model():
    return server_status.MODEL_ENCODE

@server_app.route('/req_train',methods=['POST'])
def req_train():
    name = pickle.loads(request.data)
    server_status._instance_lock.acquire()
    train_tag = req_train_core(name,server_status)
    server_status._instance_lock.release()
    
    return train_tag




import shutil

def req_train_core(name,server_status:Server_Status):
    train_tag = {
        'train':False,
        'cuda':-1
    }
    
    if name in server_status.ROUND_NAMES:
        server_status.ROUND_NAMES.remove(name)
        server_status.TRAINING_NAMES.append(name)
        cuda = server_status.CUDA_LIST.pop()
        train_tag['train'],train_tag['cuda'] = True,cuda
    return pickle.dumps(train_tag)
    
def init_info_core(info_path,server_status:Server_Status):
    with open(info_path,"r") as f:
        info = json.loads(f.read())
    server_status.DATA_INFO = info


def init_savepath_core(savepath,server_status:Server_Status):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    server_status.SAVE_PATH = savepath
    
def init_model_core(server_status:Server_Status):
    model = Model(num_classes=10)
    server_status.MODEL_ENCODE = pickle.dumps(model)

def init_names_core(server_status:Server_Status):
    server_status.DATA_NAMES = list(server_status.DATA_INFO.keys())

def init_parallel_number_core(parallel_num,server_status:Server_Status):
    server_status.PARALLEL_NUM = parallel_num
    
def send_model_core(model, name,cuda,server_status:Server_Status):
    asyn_round_core(model, name,cuda,server_status)
    
    
import random    
def asyn_round_core(model,name,cuda,server_status:Server_Status):
    main_model = pickle.loads(server_status.MODEL_ENCODE)
    server_status.ROUND+=1
    server_status.MODEL_ENCODE = pickle.dumps(aggregate_core(main_model,model))
    server_status.TRAINING_NAMES.remove(name)
    server_status.CUDA_LIST.append(cuda)
    train_able_pool = [name for name in server_status.DATA_NAMES if name not in server_status.TRAINING_NAMES]
    sample_num = server_status.PARALLEL_NUM-len(server_status.TRAINING_NAMES)-len(server_status.ROUND_NAMES)
    server_status.ROUND_NAMES = server_status.ROUND_NAMES+random.sample(train_able_pool,sample_num)
    
    
def aggregate_core(server_model:nn.Module,local_model:nn.Module,coe=0.5)->nn.Module:
    result_model = Model(num_classes=10)
    server_model:nn.Module = server_model.cpu()
    local_model = local_model.cpu()
    dictKeys = result_model.state_dict().keys()
    state_dict = {}
    for key in dictKeys:
        state_dict[key] = server_model.state_dict()[key]*(1-coe) + local_model.state_dict()[key]*coe
    result_model.load_state_dict(state_dict)
    return result_model

def init_round_names_core(server_status:Server_Status):
    server_status.ROUND_NAMES = random.sample(server_status.DATA_NAMES,server_status.PARALLEL_NUM)

def init_test_root_core(path,server_status:Server_Status):
    server_status.TEST_DATA_PATH = path
    
def init_class_num_core(class_num,server_status:Server_Status):
    server_status.CLASS_NUM = class_num

def init_cuda_core(cuda,server_status:Server_Status):
    server_status.CUDA = cuda
    server_status.CUDA_LIST = [server_status.CUDA]*server_status.PARALLEL_NUM

def init_server(args):
    server_status = Server_Status()
    init_info_core(args.info,server_status)
    init_savepath_core(args.savepath,server_status)
    init_test_root_core(args.testroot,server_status)
    init_parallel_number_core(int(args.parallelnum),server_status)
    init_class_num_core(int(args.classnum),server_status)
    init_cuda_core(int(args.cuda), server_status)
    init_names_core(server_status)
    init_model_core(server_status)
    init_round_names_core(server_status)
    


#########################################################
#####             Framework experiment tools        #####
#########################################################
def test(server_status:Server_Status,model=None):

    model = pickle.loads(server_status.MODEL_ENCODE)
    loader = test_loader_build_core(server_status)
    with torch.no_grad():
        acc = test_core(model,loader,server_status.CUDA,server_status.SAVE_PATH)
    test_log_core(acc,server_status)


from data import CIFAR_10_Dataset
def test_loader_build_core(server_status:Server_Status):
    dataset = CIFAR_10_Dataset(server_status.TEST_DATA_PATH)
    loader = DataLoader(dataset, batch_size=500, shuffle=True,pin_memory=True)
    return loader

def test_core(model:nn.Module,loader,cuda,save_path):
    acc = 0
    num = 0
    model = model.cuda(cuda)
    for batch_num,(image,label) in enumerate(loader):
        num += 1
        image = image.cuda(cuda)
        label = label.cuda(cuda)
        output = model(image)
        acc+= compute_accuracy(output,label)
    acc = acc/num
    torch.save(model,f'{save_path}/result.pth')
    return acc

def test_log_core(acc,server_status:Server_Status):
    testLog = logging.getLogger('test')
    if acc>server_status.MAX_ACC:
        server_status.MAX_ACC = acc
    testLog.error("Round: {:d}, Max accuracy: {:.2f}%, Current accuracy: {:.2f}%".format(
        server_status.ROUND, server_status.MAX_ACC*100, acc*100
    ))

import torch
def compute_accuracy(possibility, label):
    sample_num = label.size(0)
    _, index = torch.max(possibility, 1)
    correct_num = torch.sum(label == index)
    return (correct_num/sample_num).item()

import os
def check_network():
    while(True):
        time.sleep(0.01)
        if Server_Status.RECV == 0:
            Server_Status.RECV,Server_Status.SENT = check_network_core()
            if os.path.exists("log_files/sent_log.txt"):
                os.remove("log_files/sent_log.txt")
            if os.path.exists("log_files/recv_log.txt"):
                os.remove("log_files/recv_log.txt")
        else:
            recv,sent = check_network_core(Server_Status.RECV,Server_Status.SENT)
            with open("log_files/sent_log.txt", "a") as f:
                f.write(f"{sent}\n")
            with open("log_files/recv_log.txt", "a") as f:
                f.write(f"{recv}\n")
import psutil
def check_network_core(begin_recv=0,begin_sent=0):

    current_bytes_sent = psutil.net_io_counters().bytes_sent - begin_sent
    current_bytes_recv = psutil.net_io_counters().bytes_recv - begin_recv
    
    return current_bytes_recv,current_bytes_sent
                
#########################################################
#####        Framework experiment tools end         #####
#########################################################
    

