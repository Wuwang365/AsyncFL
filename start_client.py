import subprocess
import os
import time
import json




import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--info')
parser.add_argument('--port')
parser.add_argument('--ip',default='127.0.0.1')
parser.add_argument('--dataroot')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.info,'r') as f:
        names = json.loads(f.read()).keys()

    for name in names:
        cmd = f"python client_wapper.py --name {name} --dataroot {args.dataroot} --port {args.port} --ip {args.ip}"
        cmd = cmd.split(' ')
        subprocess.Popen(cmd, shell=False)
        time.sleep(0.1)