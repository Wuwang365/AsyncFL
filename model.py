from turtle import forward
import torch.nn as nn
import torch.nn.functional as F

# class SimpleCNN(nn.Module):
#     def __init__(self, numClass=10):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc = nn.Linear(32 * 8 * 8, numClass)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.relu1(out)
#         out = self.pool1(out)
#         out = self.conv2(out)
#         out = self.relu2(out)
#         out = self.pool2(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out


class ResBlock(nn.Module):
    def __init__(self,inchannels,outchannels,stride=1):
        super().__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(inchannels,outchannels,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels,outchannels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(outchannels)
        )
        self.shortcut = nn.Sequential()
        if inchannels!=outchannels or stride!=1:
            self.shortcut.add_module("shortcut",nn.Conv2d(inchannels,outchannels,kernel_size=1,stride=stride))
        
    def forward(self,x):
        out = self.convBlock(x)
        out += self.shortcut(x)
        return F.relu(out)
# 定义ResNet模型
import torch
class ResNet34(nn.Module):
    def __init__(self,numClass=128):
        super().__init__()
        self.pool = nn.MaxPool2d(stride=2,kernel_size=3,padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer1 = ResBlock(64,64,1)
        self.layer2 = nn.Sequential(ResBlock(64,128,2),ResBlock(128,128,1))
        self.layer3 = nn.Sequential(ResBlock(128,256,2),ResBlock(256,256,1))
        self.layer4 = nn.Sequential(ResBlock(256,512,2),ResBlock(512,512,1))
        self.fc = nn.Linear(512,numClass)
        
    def forward(self,x,c=None):
        x1 = self.conv1(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = F.avg_pool2d(x5,2)
        x7 = x6.view(x6.size(0),-1)
        output = self.fc(x7)
        if c==None:
            return output
        else:
            x = F.normalize(output)
            return -torch.sum((x-c)**2,dim=1)
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.mse = nn.MSELoss()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=7),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        error = torch.abs(x-x2)
        error = torch.sum(error,dim=[1,2,3])
        return error

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(512)
#         self.fc1 = nn.Linear(2048, 1024)
#         self.fc2 = nn.Linear(1024, 256)
#         self.class_2 = nn.Linear(1024, 2)
#         self.dropout = nn.Dropout(0.5)
#         self.relu = nn.ReLU()
    
#     def forward(self, x, train=False):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = F.max_pool2d(x, 2)
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = F.max_pool2d(x, 2)
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = F.max_pool2d(x, 2)
#         x = self.relu(self.bn4(self.conv4(x)))
#         x = F.max_pool2d(x, 2)
#         x = x.view(x.size(0), -1)
#         fc1 = self.relu(self.fc1(x))
#         fc1 = self.dropout(fc1)
#         fc2 = self.fc2(fc1)
#         if not train:
#             return fc2
#         result = F.softmax(self.class_2(fc1), dim=1)
#         return result 
    
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.class_2 = nn.Linear(512,2)
        self.relu = nn.ReLU()
    
    def forward(self, img, train=False):
        x = F.relu(self.conv1(img))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        fc1 = F.relu(self.fc1(x))
        fc2 = self.fc2(fc1)
        if not train:
            return fc2
        
        result = self.class_2(fc1)
        return result
    
    




import torchvision.models as models
class CustomMobileNet(nn.Module):
    def __init__(self):
        super(CustomMobileNet, self).__init__()
        # 加载预训练的 MobileNetV2 模型
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        # 获取模型的分类头的输入特征数
        num_features = self.mobilenet.classifier[1].in_features
        
        # 替换分类头的全连接层
        self.mobilenet.classifier[1] = nn.Linear(num_features, 32)
    
    def forward(self, x,c=None):
        x = self.mobilenet(x)
        if c==None:
            return x
        else:
            x = F.normalize(x, p=2, dim=1)
            c = F.normalize(c, p=2, dim=1).expand_as(x)
            return -torch.sum((x-c)**2,dim=1)
        