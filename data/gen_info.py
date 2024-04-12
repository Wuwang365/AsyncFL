import glob
import json
import random

paths = glob.glob("traindata/*")
clients = [path.split('/')[-1] for path in paths]
info = {}
delay = [0]
for client in clients:
    info[client] = {
        "delay":random.choice(delay)
    }
    
with open('info.json','w') as f:
    f.write(json.dumps(info,indent=4))