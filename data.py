from torch.utils.data import Dataset
import glob
from PIL import Image
import numpy as np
import torch
import random
    
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
        names = glob.glob(f'{path}/*/*.bmp')
        random.shuffle(names)
        for name in names:
            label = int(name.split('/')[-2])
            label = torch.tensor(label)
            img = Image.open(name)
            img = np.asarray(img)/255.0
            img = torch.as_tensor(img,dtype=torch.float)
            img = img.permute(2,0,1)
            self.data.append((img,label))

import torchvision.transforms as transforms
class KL_Dataset(Dataset):
    def __init__(self,path):
        super().__init__()
        self.data = []
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.load_data(path=path)
        
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return self.data[index]
    def load_data(self,path):
        names = glob.glob(f'{path}/*/*.bmp')
        for name in names:
            img = Image.open(name)
            img = np.asarray(img)/255.0
            img = torch.as_tensor(img,dtype=torch.float)
            img = img.permute(2,0,1)
            self.data.append(img)
            