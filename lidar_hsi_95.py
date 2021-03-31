# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import random
"""
Some tools function listed below:
"""
def set_random_seed(mySeed=0):
  torch.manual_seed(mySeed)
  torch.cuda.manual_seed(mySeed)
  torch.cuda.manual_seed_all(mySeed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(mySeed)
  random.seed(mySeed)


def max_min_mean(img):
  """
    calculate the max value ,min value ,mean value from the image.
  """
  print('max: ',np.max(img),'min: ',np.min(img),'mean: ',np.mean(img))

def map_to_255(img):
  """
    map the image to [0,255]
  """
  return ( img - np.min(img) ) / ( np.max(img)-np.min(img) ) * 255

def applyPCA(X, numComponents):
  """
    apply PCA to the image to reduce dimensionality 
  """
  newX = np.reshape(X, (-1, X.shape[2]))
  pca = PCA(n_components=numComponents, whiten=True)
  newX = pca.fit_transform(newX)
  newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
  return newX

def addZeroPadding(X, margin=2):
  """
    add zero padding to the image
  """
  newX = np.zeros((
      X.shape[0] + 2 * margin,
      X.shape[1] + 2 * margin,
      X.shape[2]
            ))
  newX[margin:X.shape[0]+margin, margin:X.shape[1]+margin, :] = X
  return newX


def createImgCube(X ,gt ,pos:list ,windowSize=25):
  """
    create Cube from pos list
    return imagecube gt nextPos
  """
  margin = (windowSize-1)//2
  zeroPaddingX = addZeroPadding(X, margin=margin)
  dataPatches = np.zeros((pos.__len__(), windowSize, windowSize, X.shape[2]))
  if( pos[-1][1]+1 != X.shape[1] ):
    nextPos = (pos[-1][0] ,pos[-1][1]+1)
  elif( pos[-1][0]+1 != X.shape[0] ):
    nextPos = (pos[-1][0]+1 ,0)
  else:
    nextPos = (0,0)
  return np.array([zeroPaddingX[i:i+windowSize, j:j+windowSize, :] for i,j in pos ]),\
  np.array([gt[i,j] for i,j in pos]) ,\
  nextPos


def createPos(shape:tuple, pos:tuple, num:int):
  """
    creatre pos list after the given pos
  """
  if (pos[0]+1)*(pos[1]+1)+num >shape[0]*shape[1]:
    num = shape[0]*shape[1]-( (pos[0])*shape[1] + pos[1] )
  return [(pos[0]+(pos[1]+i)//shape[1] , (pos[1]+i)%shape[1] ) for i in range(num) ]

def createPosWithoutZero(gt):
  """
    creatre pos list without zero labels
  """
  mask = gt>0
  return [(i,j) for i , row  in enumerate(mask) for j , row_element in enumerate(row) if row_element]

def splitTrainTestSet(X, gt, testRatio, randomState=456):
  """
    random split data set
  """
  X_train, X_test, gt_train, gt_test = train_test_split(X, gt, test_size=testRatio, random_state=randomState, stratify=gt)
  return X_train, X_test, gt_train, gt_test

def createImgPatch(lidar, pos:list, windowSize=25):
  """
    return lidar Img patches
  """
  margin = (windowSize-1)//2
  zeroPaddingLidar = np.zeros((
      lidar.shape[0] + 2 * margin,
      lidar.shape[1] + 2 * margin
            ))
  zeroPaddingLidar[margin:lidar.shape[0]+margin, margin:lidar.shape[1]+margin] = lidar
  return np.array([zeroPaddingLidar[i:i+windowSize, j:j+windowSize] for i,j in pos ])

windowSize = 11
set_random_seed()
houston_gt = loadmat('./data/houston_gt.mat')['houston_gt']
houston_hsi = loadmat('./data/houston_hsi.mat')['houston_hsi']
houston_lidar = loadmat('./data/houston_lidar.mat')['houston_lidar']
houston_hsi = applyPCA(houston_hsi,30)
trainPos = np.load("./data/trainpos.npy").astype(np.int32).tolist()
trainPos = [(array[1]-1,array[0]-1) for array in trainPos]

hsiTrain_tr,labels_tr,_=createImgCube(houston_hsi,houston_gt,trainPos,windowSize=windowSize)
hsiTrain_tr = torch.from_numpy(hsiTrain_tr.transpose((0,3,1,2))).unsqueeze(1).float()

lidarTrain_tr = createImgPatch(houston_lidar, trainPos, windowSize=windowSize)
lidarTrain_tr = torch.from_numpy(lidarTrain_tr).unsqueeze(1).float()

testPos = list(set(createPosWithoutZero(houston_gt))-set(trainPos)) 

hsiTest_te,labels_te,_=createImgCube(houston_hsi,houston_gt,testPos,windowSize=windowSize)
hsiTest_te = torch.from_numpy(hsiTest_te.transpose((0,3,1,2))).unsqueeze(1).float()

lidarTest_te = createImgPatch(houston_lidar, testPos, windowSize=windowSize)
lidarTest_te = torch.from_numpy(lidarTest_te).unsqueeze(1).float()

class TrainDS(torch.utils.data.Dataset): 
    def __init__(self):
        self.len = labels_tr.shape[0]
        self.hsi = torch.FloatTensor(hsiTrain_tr)
        self.lidar = torch.FloatTensor(lidarTrain_tr)
        self.labels = torch.LongTensor(labels_tr-1)        
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.hsi[index], self.lidar[index] , self.labels[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

class TestDS(torch.utils.data.Dataset): 
    def __init__(self):
        self.len = labels_te.shape[0]
        self.hsi = torch.FloatTensor(hsiTest_te)
        self.lidar = torch.FloatTensor(lidarTest_te)
        self.labels = torch.LongTensor(labels_te-1)        
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.hsi[index], self.lidar[index] , self.labels[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len


train_loader = torch.utils.data.DataLoader(dataset=TrainDS(), batch_size=128, shuffle=True, num_workers=2)
test_loader  = torch.utils.data.DataLoader(dataset=TestDS(),  batch_size=128, shuffle=False, num_workers=2)
print("Completed")

# SENet
class SELayer(nn.Module):  
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, dim=4):
        n, c, _, _ = x.size()
        y = self.avg_pool(x).view(n, c)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class LiST(nn.Module):
  def __init__(self, channels, args=3, drop=-1):
    super(LiST, self).__init__()
    self.drop_p = drop
    self.channel = channels // 2
    self.arg = args
    self.linear_x = nn.Linear(25, 25) #128 6400
    self.linear_y = nn.Linear(25, 25)
    self.selayer = SELayer(channels)
    if drop > 0:
        self.dropout = nn.Dropout(drop)
    self.fc_x = nn.Linear(self.channel, self.channel)
    self.fc_y = nn.Linear(self.channel, self.channel)
    self.fc_xy = nn.Linear(self.channel, channels)

  def forward(self, x, y=None, f1=None):
    if self.arg==1:
        f1 = x
        b, c, h, w = x.size()
        x = x.reshape(b, c, h * w)
        y = x[:, self.channel:, :]
        x = x[:, :self.channel, :]
        value = self.selayer(f1)
    x = self.linear_x(x).transpose(1,2) # torch.Size([128, 25, 256])
    y = self.linear_y(y).transpose(1,2) # torch.Size([128, 25, 256])
    if self.arg==3:
        f1 = f1.reshape(f1.shape[0], f1.shape[1], 5, 5)
        value = self.selayer(f1).reshape(f1.shape[0], f1.shape[1], 25)
    G = self.fc_x(x) + self.fc_y(y) # torch.Size([128, 25, 256])
    M = F.sigmoid(self.fc_xy(G))
    hsi = x * M[:, :, :self.channel] # torch.Size([128, 25, 256])
    lidar = y * M[:, :, self.channel:] # torch.Size([128, 25, 256])   通道加权
    hsi_lidar = torch.cat((hsi, lidar), dim=-1).transpose(1,2) # torch.Size([128, 512, 25])
    channel_rate = F.softmax(hsi_lidar, dim=-1)
    if self.drop_p > 0:
      channel_rate = self.dropout(channel_rate)
    if self.arg==1:
        channel_rate = channel_rate.reshape(b, c, h, w)
    res = value * channel_rate  # 空间加权
    return res



# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  print("using cuda:0 as device")
else:
  print("using cpu as device")
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.lidar_step1 = nn.Sequential(
      nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=0)
      , nn.BatchNorm2d(num_features=64)
      , nn.ReLU(inplace=True)
    )
    self.lidar_step2 = nn.Sequential(
      nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=0)
      , nn.BatchNorm2d(num_features=128)
      , nn.ReLU(inplace=True)
    )
    self.lidar_step3 = nn.Sequential(
      nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=0)
      , nn.BatchNorm2d(num_features=256)
      , nn.ReLU(inplace=True)
      , LiST(256, args=1)
    )
    """    
    conv1：（1, 30, 11, 11）， 8个 9x3x3 的卷积核 ==>（8, 22, 9, 9）
    conv2：（8, 22, 9, 9）， 16个 7x3x3 的卷积核 ==>（16, 16, 7, 7）
    conv3：（16, 16, 7, 7），32个 5x3x3 的卷积核 ==>（32, 12, 5, 5）

    """
    self.hsi_step1 = nn.Sequential(
      nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(9,3,3), padding=0)
      , nn.BatchNorm3d(num_features=8)
      , nn.ReLU(inplace=True)
    )
    self.hsi_step2 = nn.Sequential(
      nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(7,3,3), padding=0)
      , nn.BatchNorm3d(num_features=16)
      , nn.ReLU(inplace=True)
    )
    self.hsi_step3 = nn.Sequential(
      nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(5,3,3), padding=0)
      , nn.BatchNorm3d(num_features=32)
      , nn.ReLU(inplace=True)
    )
    self.hsi_step4 = nn.Sequential(
      nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
      , nn.BatchNorm2d(num_features=256)
      , nn.ReLU()
      , LiST(256, args=1)
    )

    self.list = LiST(512)

    self.drop1 = nn.Dropout(0.6)
    self.drop2 = nn.Dropout(0.6)
    self.drop3 = nn.Dropout(0.6)

    self.fusionlinear_1 = nn.Linear(in_features= 1280,out_features= 15)
    self.fusionlinear_2 = nn.Linear(in_features= 1280,out_features= 15)
    self.fusionlinear_3 = nn.Linear(in_features= 2560, out_features= 15)
    self.weight = nn.Parameter(torch.ones(2))

  def forward(self,hsi,lidar):
    x1 = self.lidar_step1(lidar)
    x2 = self.lidar_step2(x1)
    x3 = self.lidar_step3(x2)
    x4 = x3.reshape(-1, x3.shape[1], x3.shape[2]*x3.shape[3])
 
    y1 = self.hsi_step1(hsi)
    y2 = self.hsi_step2(y1)
    y3 = self.hsi_step3(y2)
    y4 = y3.reshape(-1,32*12,y3.shape[3],y3.shape[4])
    y5 = self.hsi_step4(y4)
    y6 = y5.reshape(-1, y5.shape[1], y5.shape[2]*y5.shape[3])
    
    f1 = torch.cat((x4,y6),dim=1)   ## 直接将HSI与LiDAR拼接
    xy = self.list(y6, y = x4, f1 = f1) # torch.Size([128, 512, 25])

    x = F.max_pool1d(x4,kernel_size=5)
    x = x.reshape(-1, x.shape[1] * x.shape[2]) # x4
    y = F.max_pool1d(y6,kernel_size=5)
    y = y.reshape(-1, y.shape[1] * y.shape[2]) # y6
    xy_ = F.max_pool1d(xy,kernel_size=5)
    xy_ = xy_.reshape(-1, xy_.shape[1] * xy_.shape[2]) # xy -- 2560

    x = self.drop1(x)
    y = self.drop2(y)
    xy_ = self.drop3(xy_)
    output_x = self.fusionlinear_1(x)
    output_y = self.fusionlinear_2(y)
    output_xy = self.fusionlinear_3(xy_)

    weight = F.sigmoid(self.weight)
    outputs = weight[0] * output_x + weight[1] * output_y + output_xy
    return outputs


# 网络放到GPU上
net = Net().to(device)
# 设置损失函数以及优化器
from sklearn.metrics import  classification_report,accuracy_score
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# 创建 trainloader 和 testloader
total_loss = 0
for epoch in range(110):
    net.train()
    for i, (hsi, lidar, tr_labels) in enumerate(train_loader):
        hsi = hsi.to(device)
        lidar = lidar.to(device)
        tr_labels = tr_labels.to(device)
        # 优化器梯度归零
        optimizer.zero_grad()
        # 正向传播 +　反向传播 + 优化 
        output = net(hsi,lidar)
        loss = criterion(output, tr_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    #print('[Epoch: %d] [loss avg: %.4f]   [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()))
    # scheduler.step(loss)
    net.eval()
    
    count = 0
    for hsi, lidar, gtlabels in test_loader:
      hsi = hsi.to(device)
      lidar = lidar.to(device)
      outputs = net(hsi,lidar)
      outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
      if count == 0:
          y_pred_test =  outputs
          gty = gtlabels
          count = 1
      else:
          y_pred_test = np.concatenate( (y_pred_test, outputs) )#
          gty = np.concatenate( (gty, gtlabels) )
    acc1 = accuracy_score(gty,y_pred_test)  # 准确率的计算
    print('[Epoch: %d] [loss avg: %.4f]   [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()), ' acc: ', acc1)

print('Finished Training')

count = 0
# 模型测试
from sklearn.metrics import  classification_report,accuracy_score
net.eval()
for hsi, lidar, gtlabels in test_loader:
    hsi = hsi.to(device)
    lidar = lidar.to(device)
    outputs = net(hsi,lidar)
    outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
    if count == 0:
        y_pred_test =  outputs
        gty = gtlabels
        count = 1
    else:
        y_pred_test = np.concatenate( (y_pred_test, outputs) )
        gty = np.concatenate( (gty, gtlabels) )

# 生成分类报告

classification = classification_report(gty, y_pred_test, digits=4)
print(classification)
