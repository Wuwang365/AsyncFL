from time import sleep
from flask import Flask
from server import server_app
import threading
import requests
from server import init_server
import server_global_variable
from logging.config import dictConfig
import logging

def set_dictConfig(log_name):
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] : %(message)s',
        }},
        'handlers': {
            'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        },
            'test':{
                'class': 'logging.FileHandler',
                'filename':f"./log_files/{log_name}.txt",
                'formatter':'default',
            },
                },
        'root': {
            'level': 'WARNING',
            'handlers': ['wsgi','test']
        }
        })

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(21)


app.register_blueprint(server_app)

@app.route("/")
def register():
    return "hello world"

def run_app(port):
    port = int(port)
    app.run(port=port,host="0.0.0.0")
    
import torch
import os
from server_global_variable import Server_Status
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logname',default='log_files',help='log file name')
parser.add_argument('--info',default='data/info.json',help='info path')
parser.add_argument("--parallelnum",default=8,help='parallel number of training clients')
parser.add_argument("--classnum",default=10)
parser.add_argument("--testroot",default='data/test_bmp')
parser.add_argument("--cuda",default=1)
parser.add_argument("--port",default=13275)
parser.add_argument("--savepath",default='models',help = 'model save path')
parser.add_argument("--KLroot",default='data/test_bmp')

from server import test

def exp_test_wapper(server_status):
    while(True):
        sleep(3)
        test(server_status)

from server import merge_cluster

if __name__=="__main__":
    server_status = Server_Status()
    args = parser.parse_args()
    if not os.path.exists('log_files'):
        os.mkdir('log_files')
    if not os.path.exists('models'):
        os.mkdir('models')
    init_server(args)
    set_dictConfig(args.logname)
    t1 = threading.Thread(target=exp_test_wapper,args=(server_status,))
    t2 = threading.Thread(target=merge_cluster,args=(server_status,))
    t1.start()
    t2.start()
    with torch.no_grad():
        run_app(args.port)
    
    