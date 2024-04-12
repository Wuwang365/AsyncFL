import threading
from typing import Dict, List
from uuid import UUID
import torch.nn as nn

lock = threading.Lock()



import threading

class Client_Status(object):
    _instance_lock = threading.Lock()
    TRAINING_STATUS = False
    SERVER_IP = "192.168.31.242"
    # TODO: change the server ip when deployment
    SERVER_PORT = "8080"
    GLOBAL_MODEL = ""
    CUDA = 0
    DATA_LIST = []
    TEST_DATA_ROOT = "data/data_public"
    TRAIN_DATA_ROOT = "./data/traindata/"
    WEIGHT_PATH_ROOT = "weights/"
    TEST_DATA_LIST = []
    CLIENT_NAME = ""
    TRAINING_STATUS = False
    BEGIN_TRAIN = False
    
    DATA_INDEX = 0
    DELAY = 0
    DATA_INFO = "./data/info.json"
    
    ROUND = -1
    MAX_ACC = {}
    
    def __init__(self):
        pass


    def __new__(cls, *args, **kwargs):
        if not hasattr(Client_Status, "_instance"):
            with Client_Status._instance_lock:
                if not hasattr(Client_Status, "_instance"):
                    Client_Status._instance = object.__new__(cls)  
        return Client_Status._instance