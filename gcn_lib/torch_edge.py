# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
from torch import nn
import torch.nn.functional as F


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2*torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():#x:1,3136,80  y:1,196,80
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1))#1,3136,196
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)#1,3136,1
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)#1,196,1
        return x_square + xy_inner + y_square.transpose(2, 1)


def dense_knn_matrix(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    #with torch.no_grad():
    x = x.transpose(2, 1).squeeze(-1)
    batch_size, n_points, n_dims = x.shape
    ### memory efficient implementation ###
    n_part = 10000
    if n_points > n_part:
        nn_idx_list = []
        groups = math.ceil(n_points / n_part)
        for i in range(groups):
            start_idx = n_part * i
            end_idx = min(n_points, n_part * (i + 1))
            dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
            if relative_pos is not None:
                dist += relative_pos[:, start_idx:end_idx]
            _, nn_idx_part = torch.topk(-dist, k=k)
            nn_idx_list += [nn_idx_part]
        nn_idx = torch.cat(nn_idx_list, dim=1)
    else:
        #dist = pairwise_distance(x)#8,256,512 -> 8,256,256

        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        dist =  x_square + x_inner + x_square.transpose(2, 1)
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k) # b, n, k
    ######
    center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)#1,3136,80
        y = y.transpose(2, 1).squeeze(-1)#1,196,80
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())#1,3136,196
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)#1,3136,9
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)#3136 -> 1,9,3136 ([[0,1,2...3136],...,[0,1,2...3136]]) -> 1,3136,9 000000000,111111111,2222
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):#2,1,3136,9
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index

import math
class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0,epoch_MODE=0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.epoch_MODE=epoch_MODE
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
        )
        self.f_query = self.f_key

        mo_list = [self.f_key, self.f_query]
        for m in mo_list:
            # for m in self.block_down1.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x, y=None, relative_pos=None):
        if y is not None:
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            ####
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)#2,1,3136,9
        else:

            #x = F.normalize(x, p=2.0, dim=1)
            ####
            #edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos)
            k = self.k * self.dilation

            x = x.transpose(2, 1).squeeze(-1)
            batch_size, n_points, n_dims = x.shape
            ### memory efficient implementation ###
            n_part = 10000
            if n_points > n_part:
                nn_idx_list = []
                groups = math.ceil(n_points / n_part)
                for i in range(groups):
                    start_idx = n_part * i
                    end_idx = min(n_points, n_part * (i + 1))
                    dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                    if relative_pos is not None:
                        dist += relative_pos[:, start_idx:end_idx]
                    _, nn_idx_part = torch.topk(-dist, k=k)
                    nn_idx_list += [nn_idx_part]
                nn_idx = torch.cat(nn_idx_list, dim=1)
            else:
                # dist = pairwise_distance(x)#8,256,512 -> 8,256,256

                # x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
                # x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
                # dist = x_square + x_inner + x_square.transpose(2, 1)#8,256,256

                hhh = int(x.shape[1]**0.5)
                ch = x.shape[2]
                aaa = x.transpose(2, 1).reshape(x.shape[0],ch, hhh, hhh)
                query = self.f_query(aaa)
                batch_size = query.shape[0]
                key_channels=256
                query = query.view(batch_size, key_channels, -1,1)
                #query = query.permute(0, 2, 1)  # 1,4096,64 #此时query的每一行表示一个像素点的64维(通道数)特征向量
                key = self.f_key(aaa)
                key = key.view(batch_size, key_channels, -1,1)  ##1,64,4096

                dist0 = torch.matmul(query.squeeze(dim=3).permute(0, 2, 1), key.squeeze(dim=3)) #/ math.sqrt(256)
                sim_map_0 = F.softmax(dist0, dim=-1)  # 1,4096,4096 任一像素点之间的相似性矩阵

                key = F.normalize(key, p=2.0, dim=1).squeeze(dim=3)
                query = F.normalize(query, p=2.0, dim=1).squeeze(dim=3)
                dist = torch.matmul(query.permute(0, 2, 1), key)  # 1,4096,64 * 1,64,4096 -> 1,4096,4096#
                #sim_map_n = F.softmax(dist, dim=-1)

                # import matplotlib.pyplot as plt
                # aa = dist_s[0].cpu().detach().numpy()
                # plt.imshow(aa)
                # plt.show()

                #if relative_pos is not None:
                #print('use')
                dist_s = dist + relative_pos
                _, nn_idx = torch.topk(dist_s, k=k)  # b, n, k 8,256,27
            ######
            center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)#8,256,27
            edge_index = torch.stack((nn_idx, center_idx), dim=0)#2,8,256,27
        #if self.epoch_MODE==0:
            return self._dilated(edge_index), sim_map_0
#center_idx[0]:[0,0,0,0,....,0,0,0,0,]
#              [1,1,1,1,.....,1,1,1,1]
#                      .....
#              [256,256,.....256,256]
