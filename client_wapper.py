import requests
import subprocess


import pickle
import time
def req_train(serverIp,serverPort,name,dataroot):
    url = f'http://{serverIp}:{serverPort}/server/req_train'
    while(True):
        try:
            content = requests.post(url,data=pickle.dumps(name)).content
            break
        except:
            pass
    resp = pickle.loads(content)
    if resp['train']:
        cuda = resp['cuda']
        cmd = f'python client.py --name {name} --cuda {cuda} --dataroot {dataroot} --serverport {serverPort}'
        cmd = cmd.split(' ')
        subprocess.run(cmd, shell=False)
    time.sleep(1)


def url_build_core(serverIp,serverPort):
    return f'http://{serverIp}:{serverPort}/server'

    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name')
parser.add_argument('--dataroot')
parser.add_argument('--port')
parser.add_argument('--ip')


if __name__=='__main__':
    args = parser.parse_args()
    while(True):
        name = args.name
        req_train(args.ip,args.port,name,args.dataroot)
    
    

