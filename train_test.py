<<<<<<< HEAD
# coding:utf-8
import numpy as np
import copy
import time
import scipy.io as sio
from collections import Counter
from sklearn.metrics import confusion_matrix
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os

from thop import profile
from thop import clever_format

from MIFNet import Net


from utils.CreateCube import create_patch
from utils.SplitDataset import get_data_index,generate_iter
from utils.LoadData import load_data,data_stand
from utils.SaveRecord import record_output

from utils.SaveImage import draw_testresult
from utils.SaveImage import draw_allresult
from utils.SaveImage import draw_cluster

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='PU',help='dataset')
parser.add_argument('--train_ratio', default=0.01,help='ratio of train')
parser.add_argument('--patch', default=13,help='size of patch')
parser.add_argument('--epochs',default=200,help='epoch')
parser.add_argument('--batch',default=64,help='size of batch')
parser.add_argument('--lr',default=0.0005,help='learning rate')
parser.add_argument('--iter',default=10,help='Number of iteration')
parser.add_argument('--modle_name',default='NET3',help='model name')
args = parser.parse_args()

config = {
    'dataset': args.dataset,
    'train_ratio': float(args.train_ratio),
    'patch_size': int(args.patch),
    'epoch': int(args.epochs),
    'batch_size': args.batch,
    'lr': float(args.lr),
    'iter': int(args.iter),
    'model': args.modle_name,
    'weight_decay': 0.001,
    'fc_dim': 16,
    'heads': 2,
    'drop': 0.1,
}


def evaluate_accuracy(data_iter, model, criterion, device):
    acc_sum, n = 0.0, 0
    test_l_sum, batch_num = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = criterion(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            test_l_sum += loss
            batch_num += 1
            n += y.shape[0]
    model.train()
    return [acc_sum / n, test_l_sum / batch_num]  # / test_num]


def train(model, config, device, criterion, optimizer, train_loader, val_loader, early_stopping=True, early_num=20):
    epochs = config['epoch']
    dataset = config['dataset']
    model_name = config['model']
    model = model.to(device)
    loss_list = [100]  # 记录每一个epoch的loss
    early_epoch = 0  # 记录早停的epoch

    iter_start = time.time()
    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        batch_count, train_l_sum = 0, 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)  # 使用余弦退火学习率

        epoch_start = time.time()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = criterion(y_hat, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_l_sum += loss.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]  # 记录训练集的数量
            batch_count += 1  # 记录batch的数量
        scheduler.step()

        val_acc, val_loss = evaluate_accuracy(val_loader, model, criterion, device)
        loss_list.append(val_loss)

        print(
            'epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               val_loss, val_acc, time.time() - epoch_start))

        PATH = "./net_DBA.pt"
        # print('early_epoch:',early_epoch)
        # print('loss_list:',loss_list[-1])
        if early_stopping and loss_list[-2] < loss_list[-1]:
            if early_epoch == 0:
                torch.save(model.state_dict(), PATH)
            early_epoch += 1
            loss_list[-1] = loss_list[-2]
            if early_epoch == early_num:
                model.load_state_dict(torch.load(PATH))
                break
        else:
            early_epoch = 0
    print('epoch %d, loss %.4f, train acc %.3f, iter_time %.1f sec'
          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
             time.time() - iter_start))


def test(device, model, test_loader):
    count = 0
    model.eval()
    y_pred_test = 0
    y_test = 0
    y_pred_index = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs_index = model(inputs)
            outputs_index = outputs_index.detach().cpu().numpy()
            outputs = np.argmax(outputs_index, axis=1)  # 在不影响原来张量的情况下，获取张量的值
            if count == 0:
                y_pred_test = outputs
                y_test = labels
                y_pred_index = outputs_index
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
                y_pred_index = np.concatenate((y_pred_index, outputs_index))
        return y_pred_test, y_test, y_pred_index


def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


dataset_name = config['dataset']
model_name = config['model']
train_ratio = float(config['train_ratio'])
patch = config['patch_size']
batch_size = config['batch_size']
lr = config['lr']
seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341,
         1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351]

data, gt = load_data(dataset_name)
data = data_stand(data)
all_data, all_gt = create_patch(data, gt, patch_size=patch)
band = all_data.shape[-1]
classes = int(np.max(np.unique(all_gt)))
criterion = nn.CrossEntropyLoss()  # 定义损失函数

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

oa_iter = []  # 记录每一个iter的OA
aa_iter = []  # 记录每一个iter的AA_mean(所有类的AA求和取平均值)
kappa_iter = []  # 记录每一个iter的kappa
avg_aa_iter = np.zeros((config['iter'], classes))  # 记录每一个iter的AA
traing_time = []
testing_time = []

for iter in range(config['iter']):
    model = Net(num_classes=classes)

    flops, params = profile(model, inputs=(torch.randn(batch_size, 1, patch, patch, band),))
    flops, params = clever_format([flops, params], "%.4f")
    print(f"FLOPs: {flops}, Params: {params}")

    np.random.seed(seeds[iter])  # 设置随机种子
    # 划分数据集
    train_indices, val_indices, test_indices, all_indices = get_data_index(all_gt, dataset_name, classes,
                                                                           train_ratio=train_ratio)
    train_loader, val_loader, test_loader, all_loader = generate_iter(train_indices, val_indices, test_indices,
                                                                      all_indices, all_data, all_gt, batch_size,
                                                                      classes, device)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)  # 使用Adam优化器

    # 开始训练
    print("--------------------------第" + str(iter + 1) + "iter训练开始-----------------------------")
    train_start_time = time.time()
    train(model, config, device, criterion, optimizer, train_loader, val_loader)
    train_end_time = time.time()
    print("--------------------------第" + str(iter + 1) + "iter训练结束-----------------------------")

    # 开始测试

    # path = f'./weights/{dataset_name}_{model_name}_{iter}_{best_epoch}_{best_acc}.pth'
    # model.load_state_dict(torch.load(path))
    # model = model.to(device)
    PATH = "./net_DBA.pt"
    model.load_state_dict(torch.load(PATH))
    model = model.to(device)

    test_start_time = time.time()
    pre_t, label_t, _ = test(device, model, test_loader)
    test_end_time = time.time()
    OA, AA_mean, Kappa, AA = output_metric(label_t, pre_t)

    path = f'./weights/{dataset_name}_{model_name}_{iter}_{OA}.pth'
    torch.save(model.state_dict(), path)

    best_oa = 0
    if best_oa < OA:
        best_oa = OA
        best_iter = iter
        best_iter_path = f'./weights/{dataset_name}_{model_name}_{best_iter}_{best_oa}.pth'

    oa_iter.append(OA)
    aa_iter.append(AA_mean)
    kappa_iter.append(Kappa)
    avg_aa_iter[iter, :] = AA
    traing_time.append(train_end_time - train_start_time)
    testing_time.append(test_end_time - test_start_time)

print("--------" + " Training Finished-----------")

# 将每一次iter的结果保存到文件中
record_output(oa_iter, aa_iter, kappa_iter, avg_aa_iter, traing_time, testing_time,
              './records/' + str(model_name) + "_"
              + str(dataset_name) + "_" + str(config['batch_size']) + "_" + str(config['patch_size']) + "_"
              + str(config['train_ratio']) + "_" + str(round(OA, 3)) + ".txt")

record_output_excel(config['iter'], oa_iter, aa_iter, kappa_iter, avg_aa_iter, './records/' + str(model_name) + "_"
                    + str(dataset_name) + "_" + str(config['batch_size']) + "_" + str(config['patch_size']) + "_"
                    + str(config['train_ratio']) + "_" + str(round(OA, 3)) + ".xlsx")
print("--------" + " Save data to txt Finished-----------")

# 绘制最好的iter的可视化结果
print("--------" + " Draw Visual image start-----------")
model.load_state_dict(torch.load(best_iter_path))
pre_t, label_t, y_pre_index = test(device, model, test_loader)
test_OA, _, _, _ = output_metric(label_t, pre_t)
draw_testresult(gt, test_indices, pre_t, best_oa, data_name=dataset_name, model_name=model_name)
draw_cluster(label_t, y_pre_index, classes, test_OA, dataset_name, model_name)

all_pre_test, _, _ = test(device, model, all_loader)
draw_allresult(gt, all_pre_test, best_oa, data_name=dataset_name, model_name=model_name)
print("--------" + " Draw Visual image finished-----------")
=======
# coding:utf-8
import numpy as np
import copy
import time
import scipy.io as sio
from collections import Counter
from sklearn.metrics import confusion_matrix
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os

from thop import profile
from thop import clever_format

from MIFNet import Net


from utils.CreateCube import create_patch
from utils.SplitDataset import get_data_index,generate_iter
from utils.LoadData import load_data,data_stand
from utils.SaveRecord import record_output

from utils.SaveImage import draw_testresult
from utils.SaveImage import draw_allresult
from utils.SaveImage import draw_cluster

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='PU',help='dataset')
parser.add_argument('--train_ratio', default=0.01,help='ratio of train')
parser.add_argument('--patch', default=13,help='size of patch')
parser.add_argument('--epochs',default=200,help='epoch')
parser.add_argument('--batch',default=64,help='size of batch')
parser.add_argument('--lr',default=0.0005,help='learning rate')
parser.add_argument('--iter',default=10,help='Number of iteration')
parser.add_argument('--modle_name',default='NET3',help='model name')
args = parser.parse_args()

config = {
    'dataset': args.dataset,
    'train_ratio': float(args.train_ratio),
    'patch_size': int(args.patch),
    'epoch': int(args.epochs),
    'batch_size': args.batch,
    'lr': float(args.lr),
    'iter': int(args.iter),
    'model': args.modle_name,
    'weight_decay': 0.001,
    'fc_dim': 16,
    'heads': 2,
    'drop': 0.1,
}


def evaluate_accuracy(data_iter, model, criterion, device):
    acc_sum, n = 0.0, 0
    test_l_sum, batch_num = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = criterion(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            test_l_sum += loss
            batch_num += 1
            n += y.shape[0]
    model.train()
    return [acc_sum / n, test_l_sum / batch_num]  # / test_num]


def train(model, config, device, criterion, optimizer, train_loader, val_loader, early_stopping=True, early_num=20):
    epochs = config['epoch']
    dataset = config['dataset']
    model_name = config['model']
    model = model.to(device)
    loss_list = [100]  # 记录每一个epoch的loss
    early_epoch = 0  # 记录早停的epoch

    iter_start = time.time()
    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        batch_count, train_l_sum = 0, 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)  # 使用余弦退火学习率

        epoch_start = time.time()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = criterion(y_hat, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_l_sum += loss.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]  # 记录训练集的数量
            batch_count += 1  # 记录batch的数量
        scheduler.step()

        val_acc, val_loss = evaluate_accuracy(val_loader, model, criterion, device)
        loss_list.append(val_loss)

        print(
            'epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               val_loss, val_acc, time.time() - epoch_start))

        PATH = "./net_DBA.pt"
        # print('early_epoch:',early_epoch)
        # print('loss_list:',loss_list[-1])
        if early_stopping and loss_list[-2] < loss_list[-1]:
            if early_epoch == 0:
                torch.save(model.state_dict(), PATH)
            early_epoch += 1
            loss_list[-1] = loss_list[-2]
            if early_epoch == early_num:
                model.load_state_dict(torch.load(PATH))
                break
        else:
            early_epoch = 0
    print('epoch %d, loss %.4f, train acc %.3f, iter_time %.1f sec'
          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
             time.time() - iter_start))


def test(device, model, test_loader):
    count = 0
    model.eval()
    y_pred_test = 0
    y_test = 0
    y_pred_index = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs_index = model(inputs)
            outputs_index = outputs_index.detach().cpu().numpy()
            outputs = np.argmax(outputs_index, axis=1)  # 在不影响原来张量的情况下，获取张量的值
            if count == 0:
                y_pred_test = outputs
                y_test = labels
                y_pred_index = outputs_index
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
                y_pred_index = np.concatenate((y_pred_index, outputs_index))
        return y_pred_test, y_test, y_pred_index


def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


dataset_name = config['dataset']
model_name = config['model']
train_ratio = float(config['train_ratio'])
patch = config['patch_size']
batch_size = config['batch_size']
lr = config['lr']
seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341,
         1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351]

data, gt = load_data(dataset_name)
data = data_stand(data)
all_data, all_gt = create_patch(data, gt, patch_size=patch)
band = all_data.shape[-1]
classes = int(np.max(np.unique(all_gt)))
criterion = nn.CrossEntropyLoss()  # 定义损失函数

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

oa_iter = []  # 记录每一个iter的OA
aa_iter = []  # 记录每一个iter的AA_mean(所有类的AA求和取平均值)
kappa_iter = []  # 记录每一个iter的kappa
avg_aa_iter = np.zeros((config['iter'], classes))  # 记录每一个iter的AA
traing_time = []
testing_time = []

for iter in range(config['iter']):
    model = Net(num_classes=classes)

    flops, params = profile(model, inputs=(torch.randn(batch_size, 1, patch, patch, band),))
    flops, params = clever_format([flops, params], "%.4f")
    print(f"FLOPs: {flops}, Params: {params}")

    np.random.seed(seeds[iter])  # 设置随机种子
    # 划分数据集
    train_indices, val_indices, test_indices, all_indices = get_data_index(all_gt, dataset_name, classes,
                                                                           train_ratio=train_ratio)
    train_loader, val_loader, test_loader, all_loader = generate_iter(train_indices, val_indices, test_indices,
                                                                      all_indices, all_data, all_gt, batch_size,
                                                                      classes, device)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)  # 使用Adam优化器

    # 开始训练
    print("--------------------------第" + str(iter + 1) + "iter训练开始-----------------------------")
    train_start_time = time.time()
    train(model, config, device, criterion, optimizer, train_loader, val_loader)
    train_end_time = time.time()
    print("--------------------------第" + str(iter + 1) + "iter训练结束-----------------------------")

    # 开始测试

    # path = f'./weights/{dataset_name}_{model_name}_{iter}_{best_epoch}_{best_acc}.pth'
    # model.load_state_dict(torch.load(path))
    # model = model.to(device)
    PATH = "./net_DBA.pt"
    model.load_state_dict(torch.load(PATH))
    model = model.to(device)

    test_start_time = time.time()
    pre_t, label_t, _ = test(device, model, test_loader)
    test_end_time = time.time()
    OA, AA_mean, Kappa, AA = output_metric(label_t, pre_t)

    path = f'./weights/{dataset_name}_{model_name}_{iter}_{OA}.pth'
    torch.save(model.state_dict(), path)

    best_oa = 0
    if best_oa < OA:
        best_oa = OA
        best_iter = iter
        best_iter_path = f'./weights/{dataset_name}_{model_name}_{best_iter}_{best_oa}.pth'

    oa_iter.append(OA)
    aa_iter.append(AA_mean)
    kappa_iter.append(Kappa)
    avg_aa_iter[iter, :] = AA
    traing_time.append(train_end_time - train_start_time)
    testing_time.append(test_end_time - test_start_time)

print("--------" + " Training Finished-----------")

# 将每一次iter的结果保存到文件中
record_output(oa_iter, aa_iter, kappa_iter, avg_aa_iter, traing_time, testing_time,
              './records/' + str(model_name) + "_"
              + str(dataset_name) + "_" + str(config['batch_size']) + "_" + str(config['patch_size']) + "_"
              + str(config['train_ratio']) + "_" + str(round(OA, 3)) + ".txt")

record_output_excel(config['iter'], oa_iter, aa_iter, kappa_iter, avg_aa_iter, './records/' + str(model_name) + "_"
                    + str(dataset_name) + "_" + str(config['batch_size']) + "_" + str(config['patch_size']) + "_"
                    + str(config['train_ratio']) + "_" + str(round(OA, 3)) + ".xlsx")
print("--------" + " Save data to txt Finished-----------")

# 绘制最好的iter的可视化结果
print("--------" + " Draw Visual image start-----------")
model.load_state_dict(torch.load(best_iter_path))
pre_t, label_t, y_pre_index = test(device, model, test_loader)
test_OA, _, _, _ = output_metric(label_t, pre_t)
draw_testresult(gt, test_indices, pre_t, best_oa, data_name=dataset_name, model_name=model_name)
draw_cluster(label_t, y_pre_index, classes, test_OA, dataset_name, model_name)

all_pre_test, _, _ = test(device, model, all_loader)
draw_allresult(gt, all_pre_test, best_oa, data_name=dataset_name, model_name=model_name)
print("--------" + " Draw Visual image finished-----------")
>>>>>>> b1171db (first commit)
