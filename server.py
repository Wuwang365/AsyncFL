import threading
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from model import CIFAR10Net as Model
from flask import Blueprint, request
import json
import time
import pickle
from server_global_variable import Server_Status
import logging
import copy
import uuid

server_app = Blueprint("server", __name__, url_prefix="/server")

server_status = Server_Status()


@server_app.route("/req_cfg", methods=["GET"])
def req_cfg():
    config = {
        "epoch": 10,
        "lr": 1e-2,
        "batch_size": 64
    }
    return pickle.dumps(config)


@server_app.route("/send_model", methods=["POST"])
def send_model():
    client_info = pickle.loads(request.data)
    net = client_info["model"].cpu()
    name = client_info['name']
    cuda = client_info['cuda']
    feedback = client_info['feedback']
    server_status._instance_lock.acquire()
    send_model_core(net,name,cuda,feedback,server_status)
    server_status._instance_lock.release()
    return pickle.dumps("")

@server_app.route("/req_model", methods=["POST"])
def req_model():
    client_info = pickle.loads(request.data)
    name = client_info['name']
    server_status._instance_lock.acquire()
    weights = req_model_core(name,server_status)
    server_status._instance_lock.release()
    return weights

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
    model = Model(num_class=10)
    server_status.MODEL_ENCODE = pickle.dumps(model)

def init_names_core(server_status:Server_Status):
    server_status.DATA_NAMES = list(server_status.DATA_INFO.keys())

def init_parallel_number_core(parallel_num,server_status:Server_Status):
    server_status.PARALLEL_NUM = parallel_num
    
def send_model_core(model, name,cuda,feedback,server_status:Server_Status):
    asyn_round_core(model, name,cuda,feedback,server_status)
    
def req_model_core(name,server_status:Server_Status):
    if(len(server_status.CLUSTER_2_CLIENT_NAMES.keys())<server_status.CLUSTER_NUM):
        new_cluster_key = str(uuid.uuid4())
        while(new_cluster_key in server_status.CLUSTER_2_CLIENT_NAMES.keys()):
            new_cluster_key = str(uuid.uuid4())
        server_status.CLUSTER_2_CLIENT_NAMES[new_cluster_key] = [name]
        server_status.CLUSTER_2_WEIGHT[new_cluster_key] = copy.deepcopy(server_status.MODEL_ENCODE)
        return server_status.CLUSTER_2_WEIGHT[new_cluster_key]
    else:
        weight = random.choice(list(server_status.CLUSTER_2_WEIGHT.values()))
        for key in server_status.CLUSTER_2_CLIENT_NAMES.keys():
            if(name in server_status.CLUSTER_2_CLIENT_NAMES[key]):
                weight = server_status.CLUSTER_2_WEIGHT[key]
                break
        return weight


def merge_cluster(server_status:Server_Status):
    while(True):
        time.sleep(5)
        cluster_num = len(server_status.CLUSTER_2_WEIGHT.keys())
        print(f"Current cluster number is {cluster_num}")
        if cluster_num>2*server_status.CLUSTER_NUM:
            server_status._instance_lock.acquire()
            merge_cluster_core(server_status)
            # make_center_close(server_status)
            draw = Draw()
            draw.draw_relation_matrx(server_status)
            server_status._instance_lock.release()


def make_center_close(server_status:Server_Status):
    models = []
    keys = list(server_status.CLUSTER_2_WEIGHT.keys())
    for key in keys:
        models.append(copy.deepcopy(pickle.loads(server_status.CLUSTER_2_WEIGHT[key])))
    for i in range(len(models)):
        agg_list = []
        agg_list = models[0:i]+models[(i+1):]
        other_model = aggregate_list(agg_list)
        main_model = aggregate_core(other_model,models[i],0.95)
        server_status.CLUSTER_2_WEIGHT[keys[i]] = pickle.dumps(main_model)

from data import KL_Dataset
def merge_cluster_core(server_status:Server_Status):
    curr_cluster_num = len(server_status.CLUSTER_2_WEIGHT.keys())
    curr_cluster_list = list(server_status.CLUSTER_2_WEIGHT.keys())
    KL_matrix = np.ones((curr_cluster_num,curr_cluster_num))
    KL_dataset = KL_Dataset(server_status.KL_DATA)
    KL_loader = DataLoader(KL_dataset,128,num_workers=4)
    KL_temp_result_list = []
    
    for key in curr_cluster_list:
        KL_result = KL_temp_result(KL_loader,
                                    pickle.loads(server_status.CLUSTER_2_WEIGHT[key]),
                                    server_status.CUDA)
        KL_temp_result_list.append(KL_result)
    for index_1 in range(curr_cluster_num):
        for index_2 in range(curr_cluster_num):
            KL_matrix[index_1][index_2] = KL_distance(KL_temp_result_list[index_1],KL_temp_result_list[index_2])
    
    reduce_num = curr_cluster_num - server_status.CLUSTER_NUM
    merge_pairs = find_top_k_similar_KL_value_index(KL_matrix,reduce_num)
    
    for pair in merge_pairs:
        pair = set(pair)
        merge_cluster_names = [curr_cluster_list[item] for item in pair]
        models = [pickle.loads(server_status.CLUSTER_2_WEIGHT[item]) for item in merge_cluster_names]
        clients = []
        for name in merge_cluster_names:
            clients += server_status.CLUSTER_2_CLIENT_NAMES[name]
        model = pickle.dumps(aggregate_list(models))
        
        new_cluster_key = str(uuid.uuid4())
        while(new_cluster_key in server_status.CLUSTER_2_CLIENT_NAMES.keys()):
            new_cluster_key = str(uuid.uuid4())
        
        server_status.CLUSTER_2_CLIENT_NAMES[new_cluster_key] = clients
        server_status.CLUSTER_2_WEIGHT[new_cluster_key] = model
        
        for item in pair:
            server_status.CLUSTER_2_WEIGHT.pop(curr_cluster_list[item])
            server_status.CLUSTER_2_CLIENT_NAMES.pop(curr_cluster_list[item])
    for index,model in enumerate(server_status.CLUSTER_2_WEIGHT.values()):
        torch.save(pickle.loads(model),f"{server_status.SAVE_PATH}/{index}.pth")

def aggregate_list(models):
    dict = {}
    for model in models:
        for key in model.state_dict().keys():
            if key in dict:
                dict[key]+=model.state_dict()[key]
            else:
                dict[key] = model.state_dict()[key]
    for key in dict.keys():
        dict[key]= dict[key]/len(models)
    model = Model()
    model.load_state_dict(dict)
    return model
    
import numpy as np
def find_top_k_similar_KL_value_index(matrix,k):
    matrix = np.array(matrix)
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j or i >= j:
                matrix[i][j] = np.inf
    delnum = 0
    shift = 0
    while(delnum<k-2):
        flattened_indices = np.argsort(matrix.flatten())[:k+shift]
        min_coordinates = [[index // n, index % n] for index in flattened_indices]
        min_coordinates = merge_connected_pairs(min_coordinates)
        merge_set = set()
        for cluster in min_coordinates:
            merge_set = set(cluster + list(merge_set))
        delnum = len(merge_set)-len(min_coordinates)
        shift+=1
    return min_coordinates


def merge_connected_pairs(pairs):
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))

        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            root_x = self.find(x)
            root_y = self.find(y)
            if root_x != root_y:
                self.parent[root_y] = root_x
                
    if not pairs:
        return []
    num_to_tuples = {}
    for i, tup in enumerate(pairs):
        for num in tup:
            if num not in num_to_tuples:
                num_to_tuples[num] = []
            num_to_tuples[num].append(i)

    uf = UnionFind(len(pairs))
    for nums in num_to_tuples.values():
        if len(nums) > 1:
            for i in range(1, len(nums)):
                uf.union(nums[0], nums[i])

    groups = {}
    for i in range(len(pairs)):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    merged_tuples = [[pairs[idx][j] for idx in group for j in range(2)] for group in groups.values()]

    return merged_tuples
    
    
    
def KL_distance(main_result, other_result):
    result = torch.mean(torch.nn.functional.softmax(main_result, 1)*(torch.nn.functional.log_softmax(
        main_result, 1)-torch.nn.functional.log_softmax(other_result, 1))).item()
    return result
    
    
def KL_temp_result(loader,model,cuda):
    model = model.cuda(cuda)
    result = []
    for image in loader:
        image = image.cuda(cuda)
        if result==[]:
            result = model(image).detach()
        else:
            result = torch.cat([result,model(image).detach()],dim=0)
    return result
    
    
    
    
    
import random    
def asyn_round_core(model,name,cuda,feedback,server_status:Server_Status):
    server_status.FEEDBACK.append(feedback)
    for key in server_status.CLUSTER_2_WEIGHT.keys():
        cluster_model = pickle.loads(copy.deepcopy(server_status.CLUSTER_2_WEIGHT[key]))
        server_status.CLUSTER_2_WEIGHT[key] = pickle.dumps(aggregate_core(cluster_model,model,0.03))
        
    for key in server_status.CLUSTER_2_CLIENT_NAMES.keys():
        if name in server_status.CLUSTER_2_CLIENT_NAMES[key]:
            server_status.CLUSTER_2_CLIENT_NAMES[key].remove(name)
            break
    
    main_model = pickle.loads(server_status.MODEL_ENCODE)
    if is_less_than_ratio_percent(server_status.FEEDBACK,feedback,ratio=0.8):
        new_cluster_key = str(uuid.uuid4())
        while(new_cluster_key in server_status.CLUSTER_2_WEIGHT.keys()):
            new_cluster_key = str(uuid.uuid4())
        server_status.CLUSTER_2_WEIGHT[new_cluster_key] = pickle.dumps(model)
        server_status.CLUSTER_2_CLIENT_NAMES[new_cluster_key] = [name]
    else:
        key = min_distance_key(model, server_status)
        cluster_model = pickle.loads(server_status.CLUSTER_2_WEIGHT[key])
        new_cluster_model = aggregate_core(cluster_model,model,0.5)
        server_status.CLUSTER_2_WEIGHT[key] = pickle.dumps(new_cluster_model)
        server_status.CLUSTER_2_CLIENT_NAMES[key].append(name)
    
    keys = list(server_status.CLUSTER_2_CLIENT_NAMES.keys())
    for key in keys:
        if len(server_status.CLUSTER_2_CLIENT_NAMES[key])==0:
            server_status.CLUSTER_2_CLIENT_NAMES.pop(key)
            server_status.CLUSTER_2_WEIGHT.pop(key)
        
        
    server_status.ROUND+=1
    server_status.MODEL_ENCODE = pickle.dumps(aggregate_core(main_model,model))
    server_status.TRAINING_NAMES.remove(name)
    server_status.CUDA_LIST.append(cuda)
    train_able_pool = [name for name in server_status.DATA_NAMES if name not in server_status.TRAINING_NAMES]
    sample_num = server_status.PARALLEL_NUM-len(server_status.TRAINING_NAMES)-len(server_status.ROUND_NAMES)
    server_status.ROUND_NAMES = server_status.ROUND_NAMES+random.sample(train_able_pool,sample_num)

def is_less_than_ratio_percent(nums,num,ratio=0.9):
    sorted_nums = sorted(nums)

    n = len(sorted_nums)
    start_index = 0
    end_index = int(n * ratio)
    count = sum(1 for x in sorted_nums[start_index:end_index] if x < num)

    return count > 0.9 * (end_index - start_index)


import torch.nn.utils as utils
def min_distance_key(model,server_status:Server_Status):
    min_distance = torch.inf
    min_key = ""
    for key in server_status.CLUSTER_2_WEIGHT.keys():
        model2 = pickle.loads(server_status.CLUSTER_2_WEIGHT[key])
        params1 = utils.parameters_to_vector(model.parameters())
        params2 = utils.parameters_to_vector(model2.parameters())
        distance = torch.mean(torch.abs(params1 - params2))
        if min_distance>distance:
            min_distance = distance
            min_key = key
    return min_key
    
def aggregate_core(server_model:nn.Module,local_model:nn.Module,coe=0.5)->nn.Module:
    result_model = Model(num_class=10)
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

def init_KL_data_root_core(path,server_status:Server_Status):
    server_status.KL_DATA = path

def init_server(args):
    server_status = Server_Status()
    init_info_core(args.info,server_status)
    init_savepath_core(args.savepath,server_status)
    init_test_root_core(args.testroot,server_status)
    init_parallel_number_core(int(args.parallelnum),server_status)
    init_class_num_core(int(args.classnum),server_status)
    init_cuda_core(int(args.cuda), server_status)
    init_KL_data_root_core(args.KLroot,server_status)
    init_names_core(server_status)
    init_model_core(server_status)
    init_round_names_core(server_status)
    
    


#########################################################
#####             Framework experiment tools        #####
#########################################################

class Draw():
    def __init__(self):
        self.save_times = 0
        
    def draw_relation_matrx(self,server_status:Server_Status):
        print(f"There is {len(server_status.CLUSTER_2_WEIGHT.keys())} clusters after merge")
        matrix = np.zeros([120,120])
        cluster_struture = copy.deepcopy(server_status.CLUSTER_2_CLIENT_NAMES)
        for list_item in cluster_struture.values():
            for i in list_item: 
                for k in list_item:
                    i = int(i)
                    k = int(k)
                    matrix[i][k] = 1
        matrix_re = np.zeros([120,120])
        for i in range(120):
            for k in range(120):
                matrix_re[i][k] = matrix[self.relocate(i)][self.relocate(k)]
        self.save_times+=1    
        np.savetxt(f"./relation/{self.save_times}.txt", matrix_re, fmt='%d', delimiter=' ')
    
    def relocate(self,x):
        return int(x//24+(x%24)*5)

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
    

