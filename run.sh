python start_server.py --logname test --info data/info.json --parallelnum 8 --classnum 10 --testroot data/test_bmp --cuda 1 --port 13275 --savepath models&
sleep 3
python start_client.py --info data/info.json --port 13275 --dataroot data/train_bmp
# sleep 
# pkill -u user22202695 python 