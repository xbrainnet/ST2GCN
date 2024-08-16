import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import warnings

from scipy.io import loadmat
# 固定随机数种子
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import SubsetRandomSampler, DataLoader

from code_github.model import models

warnings.filterwarnings("ignore")


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 常量定义
nfeat = 49
nhid1 = 10
nhid2 = 10
nhid3 = 10
num_windows = 4 #1,2,3,4,5,6,7,8
avg_acc = 0
avg_spe = 0
avg_recall = 0
avg_f1 = 0
avg_auc = 0
pre_ten = []
label_ten = []
gailv_ten = []
kk = 10



def stest(model, datasets_test, lam,alpha):

    eval_loss = 0
    eval_acc = 0
    pre_all = []
    labels_all = []
    gailv_all = []
    pro_all = []
    # criterion = nn.CrossEntropyLoss()
    model.eval()
    for net, data_feas, label in datasets_test:
        net, data_feas, label = net.to(DEVICE), data_feas.to(DEVICE), label.to(DEVICE)
        net = net.float()
        data_feas = data_feas.float()
        label = label.long()
        outs, loss_sim, score, edu = model(net, data_feas)

        losss = F.nll_loss(outs, label) - lam * loss_sim - alpha * edu
        # 记录误差
        eval_loss += float(losss)
        # 记录准确率
        gailv, pred = outs.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / net.shape[0]
        eval_acc += acc
        pre = pred.cpu().detach().numpy()
        pre_all.extend(pre)
        label_true = label.cpu().detach().numpy()
        labels_all.extend(label_true)
        pro_all.extend(outs[:, 1].cpu().detach().numpy())
    tn, fp, fn, tp = confusion_matrix(labels_all, pre_all).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    eval_acc_epoch = accuracy_score(labels_all, pre_all)
    precision = precision_score(labels_all, pre_all)
    recall = recall_score(labels_all, pre_all)
    f1 = f1_score(labels_all, pre_all)
    my_auc = roc_auc_score(labels_all, pro_all)

    return eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all, score



log = open('ADNI_Win4_NC_ILL.txt', mode='a', encoding='utf-8')
for yu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for lr in [1e-3,5e-3, 7e-3]:
        for lam in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]:
            for alpha in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]:
                def MaxMinNormalization(x, Max, Min):
                    x = (x - Min) / (Max - Min)
                    return x



                ################################# NC  vs   ill########################################
                from torch.utils.data import Dataset, DataLoader

                m = loadmat('')  # fmri
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

                #################################    NC  vs   SMC########################################
                # from torch.utils.data import Dataset, DataLoader

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

                ################################ NC  vs   EMCI########################################
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
                # m = loadmat('')  # fmri
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


                nets_all, fmris_all = create_DFCN(fdata, 4, yu)



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



                i = 0
                test_acc = []
                test_pre = []
                test_recall = []
                test_f1 = []
                test_auc = []
                test_sens = []
                test_spec = []
                label_ten = []
                pro_ten = []
                scores = []
                dataset = Dianxian()
                train_ratio = 0.8
                valid_ratio = 0.2
                KF = KFold(n_splits=10, shuffle=True)
                for train_idx, test_idx in KF.split(dataset):
                    train_size = int(train_ratio * len(train_idx))
                    valid_size = len(train_idx) - train_size
                    train_indices, valid_indices = train_idx[:train_size], train_idx[train_size:]
                    datasets_train = DataLoader(dataset, batch_size=20, shuffle=False, sampler=SubsetRandomSampler(train_indices))
                    datasets_valid = DataLoader(dataset, batch_size=20, shuffle=False, sampler=SubsetRandomSampler(valid_indices))
                    datasets_test = DataLoader(dataset, batch_size=20, shuffle=False, sampler=SubsetRandomSampler(test_idx))

                    epoch = 300
                    losses = []
                    acces = []
                    eval_losses = []
                    eval_acces = []
                    patiences = 30
                    min_acc = 0

                    model = models(nfeat, nhid1, nhid2, nhid3, num_windows)
                    model.to(DEVICE)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 0.005
                    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
                    for e in range(epoch):
                        train_loss = 0
                        train_acc = 0
                        model.train()
                        for ot_net, cheb, label in datasets_train:
                            ot_net, cheb, label = ot_net.to(DEVICE), cheb.to(DEVICE), label.to(DEVICE)


                            ot_net = ot_net.float()
                            cheb = cheb.float()
                            label = label.long()
                            out, loss_sim, score, edu = model(ot_net, cheb)  # torch.Size([4, 3])

                            loss = F.nll_loss(out, label) - lam * loss_sim - alpha * edu
                            # loss = criterion(out,label)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            train_loss += float(loss)
                            _, pred = out.max(1)
                            num_correct = (pred == label).sum()
                            acc = num_correct / ot_net.shape[0]
                            train_acc += acc
                        # scheduler.step()
                        plt.show()

                        losses.append(train_loss / len(datasets_train))
                        acces.append(train_acc / len(datasets_train))

                        eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all, score = stest(
                            model,
                            datasets_valid, lam,alpha)
                        if eval_acc_epoch > min_acc:
                            min_acc = eval_acc_epoch
                            torch.save(model.state_dict(), './latest' + str(i) + '.pth')
                            print("Model saved at epoch{}".format(e))

                            # min_acc = eval_acc_epoch
                            # pre_gd = precision
                            # recall_gd = recall
                            # f1_gd = f1
                            # auc_gd = my_auc
                            # sens_gd = sensitivity
                            # spec_gd = specificity
                            # labels_all_gd = labels_all
                            # pro_all_gd = pro_all
                            # s_gd = score
                            patience = 0
                        else:
                            patience += 1
                        if patience > patiences:
                            break
                        eval_losses.append(eval_loss / len(datasets_test))
                        eval_acces.append(eval_acc / len(datasets_test))
                        #     print('Eval Loss: {:.6f}, Eval Acc: {:.6f}'
                        #           .format(eval_loss / len(datasets_test), eval_acc / len(datasets_test)))
                        # '''
                        print(
                            'i:{},epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f},precision : {'
                            ':.6f},recall : {:.6f},f1 : {:.6f},my_auc : {:.6f} '
                            .format(i, e, train_loss / len(datasets_train), train_acc / len(datasets_train),
                                    eval_loss / len(datasets_valid), eval_acc_epoch, precision, recall, f1, my_auc))
                    model_test = models(nfeat, nhid1, nhid2, nhid3, num_windows)
                    model_test = model_test.to(DEVICE)
                    model_test.load_state_dict(torch.load('./latest' + str(i) + '.pth'))  # 84.3750
                    eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all, score = stest(model,
                            datasets_test, lam,alpha)

                    test_acc.append(eval_acc_epoch)
                    test_pre.append(precision)
                    test_recall.append(recall)
                    test_f1.append(f1)
                    test_auc.append(my_auc)
                    test_sens.append(sensitivity)
                    test_spec.append(specificity)
                    label_ten.extend(labels_all)
                    pro_ten.extend(pro_all)

                    i = i + 1
                print("test_acc", test_acc)
                print("test_pre", test_pre)
                print("test_recall", test_recall)
                print("test_f1", test_f1)
                print("test_auc", test_auc)
                print("test_sens", test_sens)
                print("test_spec", test_spec)
                avg_acc = sum(test_acc) / kk
                avg_pre = sum(test_pre) / kk
                avg_recall = sum(test_recall) / kk
                avg_f1 = sum(test_f1) / kk
                avg_auc = sum(test_auc) / kk
                avg_sens = sum(test_sens) / kk
                avg_spec = sum(test_spec) / kk
                print("*****************************************************")
                print('acc', avg_acc)
                print('pre', avg_pre)
                print('recall', avg_recall)
                print('f1', avg_f1)
                print('auc', avg_auc)
                print("sensitivity", avg_sens)
                print("specificity", avg_spec)

                acc_std = np.sqrt(np.var(test_acc))
                pre_std = np.sqrt(np.var(test_pre))
                recall_std = np.sqrt(np.var(test_recall))
                f1_std = np.sqrt(np.var(test_f1))
                auc_std = np.sqrt(np.var(test_auc))
                sens_std = np.sqrt(np.var(test_sens))
                spec_std = np.sqrt(np.var(test_spec))
                print("*****************************************************")
                print("acc_std", acc_std)
                print("pre_std", pre_std)
                print("recall_std", recall_std)
                print("f1_std", f1_std)
                print("auc_std", auc_std)
                print("sens_std", sens_std)
                print("spec_std", spec_std)
                print("*****************************************************")

                print(label_ten)
                print(pro_ten)
