import math

import numpy as np
import torch
from thop import profile

from torch.nn import Parameter, Module, init
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torch.nn as nn
import numpy as np
from numpy import random
from scipy.io import loadmat


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        init.xavier_uniform_(self.weight, gain=math.sqrt(2.0))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nhid)
        # self.bn1 = nn.BatchNorm1d(nhid)

    def forward(self, x, adj):
        # x = F.relu(self.gc1(x, adj))
        # print(x.shape)
        x = self.gc1(x, adj)
        return x


#
# x = torch.randn(20, 20, 48).cuda()
# adj = torch.randn(20,  20, 20).cuda()
# model = GCN(48,10).cuda()
# out = model(x,adj)
# print(out.shape)
class tensor_GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, num_windows, num_node):
        super(tensor_GCN, self).__init__()
        self.nfeat = nfeat
        self.nhid1 = nhid1
        self.nhid2 = nhid2
        self.num_node = num_node
        self.num_windows = num_windows
        self.intra_Mutil_GCN = nn.ModuleList([GCN(self.nfeat, self.nhid1) for i in range(self.num_windows)])
        self.inter_Mutil_GCN = nn.ModuleList([GCN(self.nhid1, self.nhid2) for i in range(self.num_node)])

    def forward(self, x, adj):
        # print(x.shape)
        # 图内传播
        outs = []
        for i, gcn_layer in enumerate(self.intra_Mutil_GCN):
            features = x[:, i, :, :]
            adj_new = adj[:, i, :, :]
            out = gcn_layer(features, adj_new)
            outs.append(out)
        outs = torch.stack(outs, dim=1)
        outs = F.normalize(outs)
        # print(outs.shape)
        # 图间传播
        virtual_graphs = torch.tensor(
            [[0, 1, 1.0, 1.0], [1, 0, 1, 1.0], [1.0, 1.0, 0, 1.0], [1.0, 1.0, 1.0, 0]]).cuda()
        virtual_graphs = virtual_graphs.repeat(x.shape[0], 1, 1)
        outss = []
        for i, inter_gcn_layer in enumerate(self.inter_Mutil_GCN):
            features = outs[:, :, i, :]
            out1 = inter_gcn_layer(features, virtual_graphs)
            outss.append(out1)
        outss = torch.stack(outss, dim=2)
        outss = F.normalize(outss)
        return outss


DMN = torch.tensor([22, 23, 24, 25, 30, 31, 34, 35, 38, 39, 64, 65, 66, 67, 84, 85, 86, 87])  # 18,
print(DMN.shape)
FPN = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 60, 61])  # 12
print(FPN.shape)
SAN = torch.tensor([6, 7, 12, 13, 28, 29, 30, 31, 32, 33, 58, 59])  # 12
print(SAN.shape)
ATN = torch.tensor([6, 7, 10, 11, 12, 13, 16, 17, 28, 29, 32, 33, 58, 59, 60, 61, 62, 63, 80, 81, 82, 83])  # 22
print(ATN.shape)
SMN = torch.tensor([0, 1, 18, 19, 56, 57, 68, 69])  # 8
print(SMN.shape)
VN = torch.tensor([42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55])  # 14
print(VN.shape)
AN = torch.tensor([16, 17, 62, 63, 78, 79, 80, 81])  # 8
print(AN.shape)
others = torch.tensor([14, 15, 20, 21, 26, 27, 36, 37, 40, 41, 88, 89])  # 12
print(others.shape)


def sub_network(fea, dy_net):
    # print(dy_net[0][0])
    feas = []
    dy_sub_nets = []
    fea_DMN = fea[:, :, DMN[:, None], :]
    feas.append(fea_DMN)
    net_DMN = dy_net[:, :, DMN[:, None], DMN[None, :]]
    dy_sub_nets.append(net_DMN)

    fea_FPN = fea[:, :, FPN[:, None], :]
    feas.append(fea_FPN)
    net_FPN = dy_net[:, :, FPN[:, None], FPN[None, :]]
    dy_sub_nets.append(net_FPN)

    fea_SAN = fea[:, :, SAN[:, None], :]
    feas.append(fea_SAN)
    net_SAN = dy_net[:, :, SAN[:, None], SAN[None, :]]
    dy_sub_nets.append(net_SAN)

    fea_ATN = fea[:, :, ATN[:, None], :]
    feas.append(fea_ATN)
    net_ATN = dy_net[:, :, ATN[:, None], ATN[None, :]]
    dy_sub_nets.append(net_ATN)

    fea_SMN = fea[:, :, SMN[:, None], :]
    feas.append(fea_SMN)
    net_SMN = dy_net[:, :, SMN[:, None], SMN[None, :]]
    dy_sub_nets.append(net_SMN)

    fea_VN = fea[:, :, VN[:, None], :]
    feas.append(fea_VN)
    net_VN = dy_net[:, :, VN[:, None], VN[None, :]]
    dy_sub_nets.append(net_VN)

    fea_AN = fea[:, :, AN[:, None], :]
    feas.append(fea_AN)
    # net = np.corrcoef(fea)
    net_AN = dy_net[:, :, AN[:, None], AN[None, :]]
    dy_sub_nets.append(net_AN)

    fea_others = fea[:, :, others[:, None], :]
    feas.append(fea_others)
    net_others = dy_net[:, :, others[:, None], others[None, :]]
    dy_sub_nets.append(net_others)

    return feas, dy_sub_nets


# x = torch.randn(20, 4, 90, 48)
# adj = torch.randn(20, 4, 90, 90)
# fea, nets = sub_network(x, adj)
# print(nets[6][0][0])


# 计算LM1
def compute_loss(features):
    batch_size, num_win, num_nodes, _ = features.size()
    norm_features = F.normalize(features, p=2, dim=-1)  # 归一化特征向量

    loss_M1 = 0.0
    for t in range(num_win):
        win_features = norm_features[:, t, :, :]
        cos_sim = torch.matmul(win_features, win_features.transpose(1, 2))  # 计算余弦相似度
        triu_mask = torch.triu(torch.ones(num_nodes, num_nodes), diagonal=1).bool().to(features.device)

        # 使用绝对值和相似度
        sum_cos_sim = torch.abs(cos_sim[:, triu_mask]).sum(-1)  # 取绝对值后求和

        # 累加损失
        loss_M1 += sum_cos_sim.mean()  # 平均每个批次的损失

    return loss_M1


# # 假设输入数据
# batch_size = 10
# num_win = 4
# num_nodes = 5
# feat_dim = 3
# features = torch.rand(batch_size, num_win, num_nodes, feat_dim)

# 计算损失
# loss = compute_loss(features)
# print("Computed Loss:", loss)



# def subnetwork_distance_loss(features):
#     # features shape: [batch_size, num_subnetworks, num_features]
#
#     # Expand the features tensor to prepare for pairwise distance calculation
#     f1 = features.unsqueeze(2)  # Shape: [batch_size, num_subnetworks, 1, num_features]
#     f2 = features.unsqueeze(1)  # Shape: [batch_size, 1, num_subnetworks, num_features]
#
#     # Calculate the pairwise squared Euclidean distances
#     distances = torch.sum((f1 - f2) ** 2, dim=3)
#
#     # Sum all distances within each sample in the batch
#     total_distance = torch.sum(distances, dim=[1, 2])  # Sum over both subnetwork pairs
#
#     # Average over the batch
#     loss = torch.mean(total_distance)
#     return loss
#
#
# # Example usage:
# batch_size, num_subnetworks, num_features = 10, 5, 20
# features = torch.randn(batch_size, num_subnetworks, num_features)
# loss = subnetwork_distance_loss(features)
# print("Subnetwork pairwise Euclidean distance loss:", loss)


def subnetwork_distance_loss(features):
    # features shape: [batch_size, num_subnetworks, num_features]

    # Expand the features tensor to prepare for pairwise distance calculation
    f1 = features.unsqueeze(2)  # Shape: [batch_size, num_subnetworks, 1, num_features]
    f2 = features.unsqueeze(1)  # Shape: [batch_size, 1, num_subnetworks, num_features]

    # Calculate the pairwise squared Euclidean distances
    distances = torch.sum((f1 - f2) ** 2, dim=3)

    # Normalize by the number of comparisons (num_subnetworks * (num_subnetworks - 1) / 2 for each batch)
    num_comparisons = features.size(1) * (features.size(1) - 1) / 2
    total_distance = torch.sum(distances, dim=[1, 2]) / num_comparisons

    # Average over the batch
    loss = torch.mean(total_distance)
    return loss


# Example usage:
# batch_size, num_subnetworks, num_features = 10, 5, 20
# features = torch.randn(batch_size, num_subnetworks, num_features)
# loss = subnetwork_distance_loss(features)
# print("Normalized subnetwork pairwise Euclidean distance loss:", loss)


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 dim_head,
                 heads):
        super(Attention, self).__init__()
        self.to_Q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_K = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_V = nn.Linear(dim, dim_head * heads, bias=False)
        self.norm = nn.LayerNorm(dim)

    def attention(self, Q, K, V):
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=1)
        output = torch.matmul(alpha_n, V)
        output = output.sum(1)
        return output, alpha_n

    def forward(self, x):
        Q = self.to_Q(x)
        K = self.to_K(x)
        V = self.to_V(x)
        out, alpha_n = self.attention(Q, K, V)
        out = self.norm(out)
        return out, alpha_n


class models(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid3, num_windows):
        super(models, self).__init__()
        self.nfeat = nfeat
        self.nhid1 = nhid1
        self.nhid2 = nhid2
        self.nhid3 = nhid3
        self.num_windows = num_windows
        # self.bn = nn.ModuleList([nn.BatchNorm2d(self.num_windows),
        #                          nn.BatchNorm2d(self.num_windows),
        #                          nn.BatchNorm2d(self.num_windows),
        #                          nn.BatchNorm2d(self.num_windows),
        #                          nn.BatchNorm2d(self.num_windows),
        #                          nn.BatchNorm2d(self.num_windows),
        #                          nn.BatchNorm2d(self.num_windows),
        #                          nn.BatchNorm2d(self.num_windows)])
        self.bn = nn.BatchNorm2d(self.num_windows)



        self.tgcn1 = nn.ModuleList([tensor_GCN(self.nfeat, self.nhid1, self.nhid2, self.num_windows, 18),
                                    tensor_GCN(self.nfeat, self.nhid1, self.nhid2, self.num_windows, 12),
                                    tensor_GCN(self.nfeat, self.nhid1, self.nhid2, self.num_windows, 12),
                                    tensor_GCN(self.nfeat, self.nhid1, self.nhid2, self.num_windows, 22),
                                    tensor_GCN(self.nfeat, self.nhid1, self.nhid2, self.num_windows, 8),
                                    tensor_GCN(self.nfeat, self.nhid1, self.nhid2, self.num_windows, 14),
                                    tensor_GCN(self.nfeat, self.nhid1, self.nhid2, self.num_windows, 8),
                                    tensor_GCN(self.nfeat, self.nhid1, self.nhid2, self.num_windows, 12)])
        self.tgcn2 = nn.ModuleList([tensor_GCN(self.nhid2, self.nhid1, self.nhid2, self.num_windows, 18),
                                    tensor_GCN(self.nhid2, self.nhid1, self.nhid2, self.num_windows, 12),
                                    tensor_GCN(self.nhid2, self.nhid1, self.nhid2, self.num_windows, 12),
                                    tensor_GCN(self.nhid2, self.nhid1, self.nhid2, self.num_windows, 22),
                                    tensor_GCN(self.nhid2, self.nhid1, self.nhid2, self.num_windows, 8),
                                    tensor_GCN(self.nhid2, self.nhid1, self.nhid2, self.num_windows, 14),
                                    tensor_GCN(self.nhid2, self.nhid1, self.nhid2, self.num_windows, 8),
                                    tensor_GCN(self.nhid2, self.nhid1, self.nhid2, self.num_windows, 12)])

        self.f1 = nn.Flatten()
        self.linear = nn.ModuleList([nn.Linear(18 * self.num_windows * self.nhid2, self.nhid3),
                                     nn.Linear(12 * self.num_windows * self.nhid2, self.nhid3),
                                     nn.Linear(12 * self.num_windows * self.nhid2, self.nhid3),
                                     nn.Linear(22 * self.num_windows * self.nhid2, self.nhid3),
                                     nn.Linear(8 * self.num_windows * self.nhid2, self.nhid3),
                                     nn.Linear(14 * self.num_windows * self.nhid2, self.nhid3),
                                     nn.Linear(8 * self.num_windows * self.nhid2, self.nhid3),
                                     nn.Linear(12 * self.num_windows * self.nhid2, self.nhid3)])
        self.att = Attention(self.nhid3, nhid3, 1)
        self.cl1 = nn.Linear(nhid3, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.d1 = nn.Dropout(p=0.6)
        # self.cl2 = nn.Linear(32, 2)
        # self.cl2 = nn.Linear(nhid3,32)
        # self.bn2 = nn.BatchNorm1d(32)
        # self.d2 = nn.Dropout(p=0.6)
        self.cl2 = nn.Linear(32, 2)
        self.logs = nn.LogSoftmax(dim=1)

    def forward(self, x, adj):
        fea_need = []

        loss_sim = 0
        st_feature = []
        tgcn1_out = []
        fea, dy_net = sub_network(x, adj)
        # print('fea',fea[0].shape)
        for i, tgcn_layer in enumerate(self.tgcn1):
            feature = fea[i]
            # print(feature.shape)
            feature = self.bn(torch.squeeze(feature, dim=3))
            # print(feature.shape)
            net = self.bn(dy_net[i])
            # print(net.shape)
            fea1 = tgcn_layer(feature, net)
            fea1 = self.bn(fea1)
            # print(out1.shape)
            tgcn1_out.append(fea1)

        for i, tgcn_layer in enumerate(self.tgcn2):
            net = self.bn(dy_net[i])

            out1 = tgcn_layer(tgcn1_out[i], net)
            out1 = self.bn(out1)
            # print('1', out1.shape)
            fea_need.append(out1)
            loss_sim = loss_sim + compute_loss(out1)
            out2 = self.f1(out1)
            out2 = self.linear[i](out2)  # 是为了维度对齐
            st_feature.append(out2)
        st_feature = torch.stack(st_feature, dim=1) #torch.Size([20, 8, 10])
        euclidean_distance=subnetwork_distance_loss(st_feature)
        out, score = self.att(st_feature)

        score = torch.mean(score, dim=0)
        score = torch.sum(score, dim=1)
        # print(out.shape)
        # out = self.d1(self.bn1(self.cl1(out)))
        # print('1',out.shape)
        # out = out.permute(0,2,1)
        # out = self.d2(self.bn2(self.cl2(out)))
        # out = self.logs(self.cl3(out))

        out = self.logs(self.cl2(self.d1(self.bn1(self.cl1(out)))))
        # out = self.cl2(self.d1(self.bn1(self.cl1(out))))
        return out, loss_sim, score, euclidean_distance

#
x = torch.randn(20, 4, 90, 44).cuda()
adj = torch.randn(20, 4, 90, 90).cuda()
model = models(44, 10, 10, 10, 4).cuda()
out, loss_sim, score, edu = model(x, adj)
# print(out.shape)

#
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total parameters: {:.4f}M".format(total_params / 1e6))
print("Trainable parameters: {:.4f}M".format(trainable_params / 1e6))

total = sum([param.nelement() for param in model.parameters()])
print('Number of parameter: %.4fM ' % (total / 1e6))
#
flops, params = profile(model, (x, adj,))
print('flops: ', flops, 'params: ', params)
print('flops: %.6f M, params: %.6f M' % (flops / 20000000.0, params / 1000000.0))
