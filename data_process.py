
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x
#################################     NC  vs   ILL########################################
m = loadmat('D:\codespace\ICN_tensor_GCN\datasets\ADNI_NC_SMC_EMCI_New.mat')  # fmri
keysm = list(m.keys())
fdata = m[keysm[3]]
# fdata[np.isnan(fdata)] = -1
for i in range(fdata.shape[0]):
    max_t = np.max(fdata[i])
    min_t = np.min(fdata[i])
    fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
labels = m[keysm[4]][0]
for i in range(labels.shape[0]):
    if labels[i] == 2:
        labels[i] = 1

#################################     NC  vs   SMC########################################
from torch.utils.data import Dataset, DataLoader

# m = loadmat('')
# keysm = list(m.keys())
# fdata = m[keysm[3]][0:143]
# labels = m[keysm[4]][0][0:143]  # Counter({0: 114, 1: 103, 2: 89})
# for i in range(labels.shape[0]):
#     if labels[i] == 2:
#         labels[i] = 1
# fdata[np.isnan(fdata)] = -1
# print(len(labels))
# for i in range(143):
#     max_t = np.max(fdata[i])
#     min_t = np.min(fdata[i])
#     fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)

################################    NC  vs   EMCI########################################
# from torch.utils.data import Dataset, DataLoader
# #
# # # 73,70,60
#
# m = loadmat('')  # fmri
# keysm = list(m.keys())
# fdata = m[keysm[3]]
# fdata = torch.cat((torch.tensor(fdata[0:73]), torch.tensor(fdata[143:203])))
# fdata = fdata.numpy()
#
# labels = m[keysm[4]][0]
# labels = torch.cat((torch.tensor(labels[0:73]), torch.tensor(labels[143:203])))
# labels = labels.numpy()
# for i in range(labels.shape[0]):
#     if labels[i] == 2:
#         labels[i] = 1

#################################  SMC  vs   EMCI########################################
# from torch.utils.data import Dataset, DataLoader
#
# m = loadmat(')  # fmri
# keysm = list(m.keys())
# fdata = m[keysm[3]]
# fdata = fdata[73:203]
# for i in range(fdata.shape[0]):
#     max_t = np.max(fdata[i])
#     min_t = np.min(fdata[i])
#     fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
# labels = m[keysm[4]][0]
# labels = labels[73:203]
# for i in range(labels.shape[0]):
#     if labels[i] == 2:
#         labels[i] = 0
# for i in range(labels.shape[0]):
#     if labels[i] == 2:
#         labels[i] = 1
# 对应打乱数据集
index = [i for i in range(fdata.shape[0])]
np.random.shuffle(index)
fdata = fdata[index]
labels = labels[index]


#######################################创建动态脑网络##########################################
def create_DFCN(dataset, num_window, yuzhi):
    nets_all = []
    fmris_all = []
    win_length = dataset.shape[2] // num_window
    for i in range(dataset.shape[0]):
        nets = []
        fmri_sub = []
        datas = dataset[i]  # 90*240
        for j in range(num_window):
            window = datas[:, win_length * j:win_length * (j + 1)]
            fmri_sub.append(window)
            net = np.corrcoef(window)
            net = np.abs(net)
            nets.append(net)
        nets_all.append(nets)
        fmris_all.append(fmri_sub)
    nets_all = np.array(nets_all)
    fmris_all = np.array(fmris_all)
    nets_all[nets_all < yuzhi] = 0
    return nets_all, fmris_all  # torch.Size([306, 4, 90, 90])


nets_all, fmris_all = create_DFCN(fdata, 4, 0.5)


class Dianxian(Dataset):
    def __init__(self):
        super(Dianxian, self).__init__()
        self.feas = fmris_all
        self.nets = nets_all
        self.label = labels

    def __getitem__(self, item):
        fea = self.feas[item]
        net = self.nets[item]
        label = self.label[item]
        return fea, net, label

    def __len__(self):
        return self.feas.shape[0]

