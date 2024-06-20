# python start_server.py --logname test --info data/info.json --parallelnum 8 --classnum 10 --testroot data/testdata --cuda 3 --port 25589 --savepath models&
# sleep 3
# python start_client.py --info data/info.json --port 25589 --dataroot data/non_cluster/beta_0.1
# sleep 3
# pkill -u user22202695 python 

python start_server.py --logname test --info data/info.json --parallelnum 8 --classnum 100 --testroot data/cifar100/0 --cuda 3 --port 25589 --savepath models&
sleep 3
python start_client.py --info data/info.json --port 25589 --dataroot data/cifar100/beta_0.1