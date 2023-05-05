"""
内容:复数胶囊网络的实现与融合特征的分类，以及最终的complex-caspnet性能测试
    代码原始架构参考链接:https://github.com/Riroaki/CapsNet
"""
import torch
from torch import nn
from torch.autograd import Variable

from torch.nn import functional as F

import numpy as np
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

import sys
user = sys.argv[1]

def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype)+1j*(fr(input.imag)+fi(input.real)).type(dtype)

class ComplexConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    def forward(self,input):
        return apply_complex(self.conv_r, self.conv_i, input)

def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    scale=torch.tanh(squared_norm.sqrt())
    return scale * x / (squared_norm.sqrt() + 1e-8)

from torch.nn.functional import relu
def complex_relu(input):
    return relu(input.real).type(torch.complex64)+1j*relu(input.imag).type(torch.complex64)

class PrimaryCaps(nn.Module):
    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()
        self.conv = ComplexConv2d(in_channels=in_channels,
                              out_channels=out_channels * num_conv_units,
                              kernel_size=kernel_size,
                              stride=stride)
        self.out_channels = out_channels

    def forward(self, x):
        out = self.conv(x)
        batch_size = out.shape[0]
        return squash(out.contiguous().view(batch_size, -1, self.out_channels), dim=-1)

class DigitCaps(nn.Module):
    def __init__(self, in_dim, in_caps, out_caps, out_dim, num_routing):
        super(DigitCaps, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.num_routing = num_routing
        self.device = device
        self.W = nn.Parameter(0.01 * torch.randn(1, out_caps, in_caps, out_dim, in_dim),
                              requires_grad=True).type(torch.complex64)#.cuda()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1).unsqueeze(4)
        u_hat = torch.matmul(self.W, x)
        u_hat = u_hat.squeeze(-1)
        temp_u_hat = u_hat.detach()
        b = torch.zeros(batch_size, self.out_caps, self.in_caps, 1)#.to(self.device)
        for route_iter in range(self.num_routing-1):
            b=b.type(torch.float64)
            c = b.softmax(dim=1)
            s = (c * temp_u_hat).sum(dim=2)
            v = squash(s).type(torch.complex64)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b = uv
        b=b.real**2+b.imag**2
        c = b.softmax(dim=1)
        u_hat=u_hat.type(torch.float64)
        s = (c * u_hat).sum(dim=2)
        v = squash(s)
        return v

class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv = ComplexConv2d(2, 256, 9)
        self.primary_caps = PrimaryCaps(num_conv_units=32, in_channels=256, out_channels=8, kernel_size=8, stride=2)
        self.digit_caps = DigitCaps(in_dim=8, in_caps=96, out_caps=2, out_dim=16, num_routing=3)
        self.decoder = nn.Sequential(
            nn.Linear(16 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 640),
            nn.Sigmoid())

    def forward(self, x):
       # x=x.cuda()
        out = complex_relu(self.conv(x))
        out = self.primary_caps(out)
        out = self.digit_caps(out)
        logits = torch.norm(out, dim=-1)
        pred = torch.eye(2).to(device).index_select(dim=0, index=torch.argmax(logits, dim=1))
        batch_size = out.shape[0]
        reconstruction = self.decoder((out * pred.unsqueeze(2)).contiguous().view(batch_size, -1).type(torch.float32))
        return logits, reconstruction

class CapsuleLoss(nn.Module):
    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.reconstruction_loss_scalar = 5e-9
        self.mse = nn.MSELoss(reduction='sum')
    def forward(self, images, labels, logits, reconstructions):
        left = (self.upper - logits).relu() ** 2
        right = (logits - self.lower).relu()

        margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)
        reconstruction_loss = self.mse(reconstructions.contiguous().view(images.shape), images.type(torch.float32).cpu())
        return margin_loss + self.reconstruction_loss_scalar * reconstruction_loss
"""
导入融合特征和多模态分类器
"""
import os
test_features=[]
fuse_path = os.path.join(os.path.join("EATD-Corpus", user), 'fuse_features.npy')
test_features.append(fuse_path)

caspnet_list=["Detect/Model/caspnetcomplex1_0.90_2.pt"]
test_feature=np.load(test_features[0])
train=np.load(test_features[0])
model=torch.load(caspnet_list[0],map_location='cpu')

x_test= Variable(torch.from_numpy(test_feature).type(torch.complex64), requires_grad=True)
x_test=x_test.reshape(-1, 2,20,16)
logits, reconstructions = model(x_test)
pred_labels = torch.argmax(logits, dim=1)
pred_labels=pred_labels.cpu().detach().numpy()
s = F.softmax(logits)

import os
import pandas as pd

# 创建用户的csv文件
if os.path.isfile("EATD-Corpus/"+str(user)+"/"+"record.csv")==False:
    data=pd.DataFrame(columns=["序号","抑郁概率","正常概率","结果","SDS分数上限","SDS分数下限","预测SDS分数","sds_x","sds_y","个人摘要"])
    data.to_csv("EATD-Corpus/"+str(user)+"/"+"record.csv",index=False)

import datetime
today=datetime.date.today()
timelist=[]
prolist=[]
normlist=[]
resultslist=[]

for id,i in enumerate(pred_labels):
    if i == 1:
        b, a = s.cpu().detach().numpy()[id][0], round(s.cpu().detach().numpy()[id][1],4)
        print("存在抑郁倾向" )
        prolist = a
        normlist = b
        resultslist = i
    else:
        b, a = round(s.cpu().detach().numpy()[id][0],4), s.cpu().detach().numpy()[id][1]
        print("情绪状态正常")
        prolist = a
        normlist = b
        resultslist = i

import pandas as pd
import os

fold_num = sum([os.path.isdir("EATD-Corpus/"+str(user)+"/"+listx) for listx in os.listdir("EATD-Corpus/"+str(user))])

data=pd.read_csv("EATD-Corpus/"+str(user)+"/"+"record.csv")

flist=np.concatenate([data["序号"].values,[fold_num]])
plist=np.concatenate([data["抑郁概率"].values,[prolist]])
nlist=np.concatenate([data["正常概率"].values,[normlist]])
rlist=np.concatenate([data["结果"].values,[resultslist]])
data.loc[len(data)]=np.nan

data["序号"]=flist
data["抑郁概率"]=plist
data["正常概率"]=nlist
data["结果"]=rlist
data.to_csv("EATD-Corpus/"+str(user)+"/"+"record.csv",index=False)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

# In[]
xgb=XGBRegressor()
lgb=LGBMRegressor()
xgb = joblib.load('Detect/Model/xgb.dat')
lgb = joblib.load("Detect/Model/lgb.dat")
# In[]
"""
预测分数值
"""
pred=np.zeros(train.shape[0])
for i in range(train.shape[0]):
    a,b=s.cpu().detach().numpy()[i][0],s.cpu().detach().numpy()[i][1]
    s1=xgb.predict(train[i].reshape(1,-1))[0]
    s2=lgb.predict(train[i].reshape(1,-1))[0]
    pred[i]=max(s1,s2)*b+min(s1,s2)*a
    a1=max(s1,s2)
    a2=min(s1,s2)
    maxlist=a1
    minlist=a2

# print("对用户"+str(user)+"的抑郁测试结果为:")
for id,i in enumerate(pred):
    predlist = i

d={
    "SDS分数上限":maxlist,
    "SDS分数下限":minlist,
    "预测SDS分数":predlist
}

data=pd.read_csv("EATD-Corpus/"+str(user)+"/"+"record.csv")
if data.shape[0]==1:
    data["SDS分数上限"]=maxlist
    data["SDS分数下限"]=minlist
    data["预测SDS分数"]=predlist
    data.to_csv("EATD-Corpus/"+str(user)+"/"+"record.csv",index=False)

else:
    flist=np.concatenate([data["SDS分数上限"].dropna().values,[maxlist]])
    plist=np.concatenate([data["SDS分数下限"].dropna().values,[minlist]])
    nlist=np.concatenate([data["预测SDS分数"].dropna().values,[predlist]])
    data["SDS分数上限"]=flist
    data["SDS分数下限"]=plist
    data["预测SDS分数"]=nlist
    data.to_csv("EATD-Corpus/"+str(user)+"/"+"record.csv",index=False)
