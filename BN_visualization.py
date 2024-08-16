import numpy as np
import openpyxl
from matplotlib import pyplot as plt
from scipy import stats
from scipy.io import loadmat



m = loadmat('D:\codespace\ICN_tensor_GCN\sub_ICN_tensor_gcn\TLE0.250.010.01.mat')
keysm = list(m.keys())
print(keysm)
fea1 = m['feas1']
fea2 = m['feas2']
fea3 = m['feas3']
fea4 = m['feas4']
fea5 = m['feas5']
fea6 = m['feas6']
fea7 = m['feas7']
fea8 = m['feas8']

# print(fea1[0][0])
# print(fea1[1][0].shape)
labels = m['label'][0]
my_list = list(labels)
select_sub2 = []
select_sub1 = []
target_element0 = 0

for i in range(len(my_list)):
    if my_list[i] == target_element0:
        select_sub1.append(i)
target_element1 = 1

for i in range(len(my_list)):
    if my_list[i] == target_element1:
        select_sub2.append(i)

# Brain network visualization
select_fea11 = np.mean(fea1[select_sub1, :, :, :], axis=(0, 1))
select_fea21 = np.mean(fea1[select_sub2, :, :, :], axis=(0, 1))
print('DMN', np.mean((np.abs(np.corrcoef(select_fea11))), axis=(0, 1)))
print('DMN', np.mean((np.abs(np.corrcoef(select_fea21))), axis=(0, 1)))

select_fea12 = np.mean(fea2[select_sub1, :, :, :], axis=(0, 1))
select_fea22 = np.mean(fea2[select_sub2, :, :, :], axis=(0, 1))
print('FPN', np.mean((np.abs(np.corrcoef(select_fea12))), axis=(0, 1)))
print('FPN', np.mean((np.abs(np.corrcoef(select_fea22))), axis=(0, 1)))

select_fea13 = np.mean(fea3[select_sub1, :, :, :], axis=(0, 1))
select_fea23 = np.mean(fea3[select_sub2, :, :, :], axis=(0, 1))
print('SAN', np.mean((np.abs(np.corrcoef(select_fea13))), axis=(0, 1)))
print('SAN', np.mean((np.abs(np.corrcoef(select_fea23))), axis=(0, 1)))

select_fea14 = np.mean(fea4[select_sub1, :, :, :], axis=(0, 1))
select_fea24 = np.mean(fea4[select_sub2, :, :, :], axis=(0, 1))
print('ATN',np.mean((np.abs(np.corrcoef(select_fea14))), axis=(0, 1)))
print('ATN',np.mean((np.abs(np.corrcoef(select_fea24))), axis=(0, 1)))

select_fea15 = np.mean(fea5[select_sub1, :, :, :], axis=(0, 1))
select_fea25 = np.mean(fea5[select_sub2, :, :, :], axis=(0, 1))
print('SMN',np.mean((np.abs(np.corrcoef(select_fea15))), axis=(0, 1)))
print('SMN',np.mean((np.abs(np.corrcoef(select_fea25))), axis=(0, 1)))

select_fea16 = np.mean(fea6[select_sub1, :, :, :], axis=(0, 1))
select_fea26 = np.mean(fea6[select_sub2, :, :, :], axis=(0, 1))
print('VN',np.mean((np.abs(np.corrcoef(select_fea16))), axis=(0, 1)))
print('VN',np.mean((np.abs(np.corrcoef(select_fea26))), axis=(0, 1)))

select_fea17 = np.mean(fea7[select_sub1, :, :, :], axis=(0, 1))
select_fea27 = np.mean(fea7[select_sub2, :, :, :], axis=(0, 1))
print('AN',np.mean((np.abs(np.corrcoef(select_fea17))), axis=(0, 1)))
print('AN',np.mean((np.abs(np.corrcoef(select_fea27))), axis=(0, 1)))

select_fea18 = np.mean(fea8[select_sub1, :, :, :], axis=(0, 1))
select_fea28 = np.mean(fea8[select_sub2, :, :, :], axis=(0, 1))
print('ON',np.mean((np.abs(np.corrcoef(select_fea18))), axis=(0, 1)))
print('ON',np.mean((np.abs(np.corrcoef(select_fea28))), axis=(0, 1)))

NC = np.concatenate(
    (select_fea11, select_fea12, select_fea13, select_fea14, select_fea15, select_fea16, select_fea17, select_fea18),
    axis=0)
ILL = np.concatenate(
    (select_fea21, select_fea22, select_fea23, select_fea24, select_fea25, select_fea26, select_fea27, select_fea28),
    axis=0)

net1 = np.corrcoef(NC)
# net[net<0.]=0
plt.imshow(net1, cmap='coolwarm', vmin=-1, vmax=1)  # coolwarm
plt.colorbar()
plt.show()

net2 = np.corrcoef(ILL)
plt.imshow(net2, cmap='coolwarm', vmin=-1, vmax=1)  # coolwarm
plt.colorbar()
plt.show()

