from torch.utils.data import Dataset
import glob
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

train_transform = transforms.Compose([ # 随机裁剪并填充
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # 归一化
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

    
class CIFAR_10_Dataset(Dataset):
    def __init__(self,path) -> None:
        super().__init__()
        self.data = []
        self.load_data(path=path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0],self.data[index][1]
    
    def load_data(self,path):
        names = glob.glob(f'{path}/*/*.png')
        for name in names:
            label = int(name.split('/')[-2])
            label = torch.tensor(label)
            img = Image.open(name)
            img = test_transform(img)
            self.data.append((img,label))
            

class Class_Wise_Dataset(Dataset):
    def __init__(self,path,target) -> None:
        super().__init__()
        self.data = []
        self.load_data(path=path,target=target)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0],self.data[index][1]
    
    def load_data(self,path,target):
        target_names = glob.glob(f'{path}/{target}/*.png')
        for name in target_names:
            img = Image.open(name)
            img_1 = train_transform(img)
            img_2 = train_transform(img)
            self.data.append((img_1,img_2))


import random
class Class_Wise_Con_Dataset(Dataset):
    def __init__(self,path,target) -> None:
        super().__init__()
        self.target_data = []
        self.non_target_data = []
        self.load_data(path=path,target=target)
        
    def __len__(self):
        return min(len(self.target_data),len(self.non_target_data))

    def __getitem__(self, index):
        return self.target_data[index][0],self.target_data[index][1],self.non_target_data[index][0],self.non_target_data[index][1]
    
    def load_data(self,path,target):
        
        target_names = glob.glob(f'{path}/{target}/*.png')
        names = glob.glob(f'{path}/*/*.png')
        random.shuffle(target_names)
        random.shuffle(names)
        count = 0
        for name in target_names:
            count+=1
            img = Image.open(name)
            img = test_transform(img)
            label = torch.tensor(0)
            self.target_data.append((img,label))
        non_count = 0
        for name in names:
            if not(name in target_names) and non_count<count:
                non_count+=1
                img = Image.open(name)
                img = test_transform(img)
                label = torch.tensor(1)
                self.non_target_data.append((img,label))