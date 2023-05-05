"""
语音和文本特征向量的融合
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

import warnings
warnings.filterwarnings("ignore")
"""
text_feature是predict文本的
audio是存语音的三类特征向量
"""

import sys
user = sys.argv[1]
print(user)

import os
text_path = os.path.join(os.path.join("EATD-Corpus", user), 'predict_text.npz')
audio_path = os.path.join(os.path.join("EATD-Corpus", user), 'feature.npz')

text_features = np.load(text_path)['arr_0']
print("text_features:")
print(text_features.shape)

audio_features = np.squeeze(np.load(audio_path)['arr_0'], axis=2)
print("audio_features:")
print(audio_features.shape)

fuse_features = [[audio_features[i], text_features[i]] for i in range(text_features.shape[0])]

class TextBiLSTM(nn.Module):
    def __init__(self, config):
        super(TextBiLSTM, self).__init__()
        self.num_classes = config['num_classes']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout']
        self.hidden_dims = config['hidden_dims']
        self.rnn_layers = config['rnn_layers']
        self.embedding_size = config['embedding_size']
        self.bidirectional = config['bidirectional']
        self.build_model()
        self.init_weight()
    def init_weight(net):
        for name, param in net.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def build_model(self):
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )
        self.lstm_net = nn.GRU(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout,
                                bidirectional=self.bidirectional)
        self.fc_out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes),
            # nn.ReLU(),
            nn.Softmax(dim=1),
        )
        self.ln1 = nn.LayerNorm(self.embedding_size)
        self.ln2 = nn.LayerNorm(self.hidden_dims)

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        lstm_hidden = lstm_hidden.unsqueeze(1)
        atten_w = self.attention_layer(lstm_hidden)
        m = nn.Tanh()(h)
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        softmax_w = F.softmax(atten_context, dim=-1)
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.ln1(x)
        output, (final_hidden_state, final_cell_state) = self.lstm_net(x)
        output = output.permute(1, 0, 2)
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        atten_out = self.attention_net_with_w(output, final_hidden_state)
        atten_out = self.ln2(atten_out)
        return self.fc_out(atten_out)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))		
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class Residual(nn.Module):
    def __init__(self, input_channels, out_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.CBAM = CBAM(out_channels)
    def forward(self, X):
        Y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(X)),inplace=True)))
        Y = self.CBAM(Y) * Y
        
        if self.conv3:
            X = self.conv3(X) 
        Y += X  
        return F.relu(Y)

def resnet_block(input_channels, out_channels, num_residuals, first_block=False):
    block = []
    for i in range(num_residuals):
        if i == 0 and not first_block: 
            block.append(Residual(input_channels, out_channels, use_1x1conv=True, strides=2))
        else:
            block.append(Residual(out_channels, out_channels))
    return block

b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

"""
文本综合模型和语音综合模型
"""
text_model_paths = ['Detect/Model/Bert_128_0.82_2.pt']
audio_model_paths = ['Detect/Model/resnet18_0.91_3.pt']

bigru=torch.load(text_model_paths[0], map_location='cpu')
resnet=torch.load(audio_model_paths[0], map_location='cpu')

x_text = []
x_audio = []

for ele in fuse_features:
    x_text.append(ele[1])
    x_audio.append(ele[0])

x_text=np.array(x_text)
x_audio=np.array(x_audio)
x_text_train, x_audio_train = Variable(torch.tensor(x_text).type(torch.FloatTensor), requires_grad=False), \
Variable(torch.tensor(x_audio).type(torch.FloatTensor), requires_grad=False)#.cuda()

print(x_text_train.shape)
x_text_train = x_text_train.permute(1, 0, 2)#.cuda()
x_text_train=bigru.ln1(x_text_train)
output, h = bigru.lstm_net(x_text_train)
output = output.permute(1, 0, 2)#.cuda()
h = h.permute(1, 0, 2)
atten_out = bigru.attention_net_with_w(output, h)
atten_out=bigru.ln2(atten_out)
x_audio_train=x_audio_train.reshape(-1,3,16,16)

for i in range(len(resnet)):
    if i==7:
        break
    x_audio_train=resnet[i](x_audio_train)#.cuda()
x_new_train=torch.cat((atten_out,x_audio_train),dim=1)

# 保存融合特征

savepath = os.path.join(os.path.join("EATD-Corpus", user), 'fuse_features')
np.save(savepath,x_new_train.cpu().detach().numpy())
print(x_new_train.shape)