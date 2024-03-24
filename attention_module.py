#coding=gbk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
import numpy as np

class _SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1,epoch_MODE=0):#(128, int(128 / 2), 128, out_channels=128, scale=1)
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels#128
        self.out_channels = out_channels#128
        self.key_channels = key_channels#128/2=64
        self.value_channels = value_channels#128
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.key_channels),
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=3, stride=1, padding=1)#128 -> 128
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1, padding=0)
        # nn.init.constant(self.W.weight, 0)
        # nn.init.constant(self.W.bias, 0)

        mo_list = [self.W,self.f_value,self.f_key,self.f_query]
        for m in mo_list:
        #for m in self.block_down1.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

        self.mm = torch.nn.AvgPool2d(3,1,1)

        self.M = np.ones((256,256))
        self.M [np.eye(256, dtype=np.bool)] = 0

        self.M = torch.Tensor(self.M).cuda()
        self.alpha = nn.Parameter(torch.tensor(
            5., dtype=torch.float32))
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / 4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / 4), in_channels)
        )

    def forward(self, x,epoch_MODE=None):

        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        value = self.f_value(x)#1,128,64,64 -> 1,128,64,64
        value = value.view(batch_size, self.value_channels, -1)#1,128,4096
        value = value.permute(0, 2, 1)#1,4096,128 #此时value的每一行表示一个像素点的128维(通道数)特征向量

        query = self.f_query(x)
        query = query.view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)#1,4096,64 #此时query的每一行表示一个像素点的64维(通道数)特征向量

        key = self.f_key(x)
        key = key.view(batch_size, self.key_channels, -1)##1,64,4096

        sim_map = torch.matmul(query, key)#1,4096,64 * 1,64,4096 -> 1,4096,4096
        sim_map = F.softmax(sim_map, dim=-1)#1,4096,4096 任一像素点之间的相似性矩阵

        context = torch.matmul(sim_map, value)#(4096,4096) * (4096,128)sim_map的第一行指的是第一个像素点与其他所有像素点的整体的相关性（归一化之后），相乘表示用全局相关性更新现有的value
        context = context.permute(0, 2, 1).contiguous()#1,128,4096
        context = context.view(batch_size, self.value_channels, *x.size()[2:])#1,128,64,64
        context = self.W(context)#1,128,64,64
        context = x+context
        return context






class AttentionModule(nn.Module):
    def __init__(self, dim,k,dilation):
        super().__init__()
        # depth-wise convolution
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # depth-wise dilation convolution
        self.conv_spatial = nn.Conv2d(dim, dim, k, stride=1, padding=int((k-1)*dilation*0.5), groups=dim, dilation=dilation)
        # channel convolution (1×1 convolution)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, dim,k,dilation):#C,3,3
        super().__init__()
        self.p1 = nn.Conv2d(dim,dim,1)
        self.ac = nn.ReLU()
        self.AT = AttentionModule(dim,k,dilation)
        self.p2 = nn.Conv2d(dim,dim,1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.AT(x)
        x = self.ac(x)
        x = self.p2(x)
        x=x+shortcut
        return x
