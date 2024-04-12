Build base on flask and torch
You need to modify ```CUDA_LIST``` and ```RAW_CUDA_LIST``` to satisfy with your cuda device
## run server
```python start_server.py```
## run client
```python start_client.py```
start_client is employed to run 120 clients parallelly.
To run one client ```python client_wapper.py```
## log file
You can find the run log at ```log_files/server_log.txt```

## example data format
```
|-data // data directory
|--traindata // traindata directory
|---1 // client 1 data directory
|----0 // client 1 class 0 directory
|-----1.png
|-----2.png
|----1
|-----3.png
```

## Start your Expert-FL
```
start server:
python start_server.py --logname {} --info {} --parallelnum {} --classnum {} --testroot {} --cuda {} --port {}

start client:
python start_client.py --info {} --port {} --dataroot {}
```
> For above data format, ```--testroot``` should be ```data/testdata```,```--dataroot``` should be ```data/traindata``` 

> Make sure your ```parallelnum``` will not result in OOM, it reasonable to set as 8 for 3090