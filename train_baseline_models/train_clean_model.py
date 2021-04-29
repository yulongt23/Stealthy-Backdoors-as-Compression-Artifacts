# Adapted from CIFAR-10 training example at https://github.com/kuangliu/pytorch-cifar

'''Train clean models or regular backdoored models
'''
import os
import sys
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

sys.path.append("..")
import gtsrb_dataset
from models import *
from utils import progress_bar
from precision_utils import add_inputs_with_trigger
from precision_utils import get_dataset_info
from precision_utils import get_normal_model

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset-name', type=str, help='the dataset to use')
parser.add_argument('--network-arch', type=str, help='the model to use')
parser.add_argument('--trigger', action='store_true', help='train backdoored model model')
parser.add_argument('--target-label', type=int, default=0, help='target label')
parser.add_argument('--version', type=int, default=0,  help='version number / random seed')
parser.add_argument('--ckpt-path', type=str, default='../checkpoint/',  help='checkpoint path')
args = parser.parse_args()


version_number = args.version
np.random.seed(version_number)
random.seed(version_number)
torch.manual_seed(version_number)
torch.cuda.manual_seed(version_number)
torch.backends.cudnn.deterministic = True

device = 'cuda'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0

dataset_name = args.dataset_name
network_arch = args.network_arch
training_batch_size = 128
is_backdoor_training = args.trigger
target_label = args.target_label

## ckpt path prefix
if is_backdoor_training:  # normal backdoor training
    dir_prefix = network_arch + '_' + dataset_name + '_square_trigger_label' + str(target_label)
else:  # clean model training
    dir_prefix = network_arch + '_' + dataset_name + '_clean_v' + str(version_number)

print('network arch:', network_arch)
print('target label:', target_label)
print('dataset name:', dataset_name)
print('dir_prefix:', dir_prefix)
print('is backdoor training:', is_backdoor_training)

if dataset_name == 'gtsrb':
    target_label = round(target_label * 43/ 10 +1)

trainset, testset = get_dataset_info(dataset_name, '../data')
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=training_batch_size, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

print('==> Building model..')

net = get_normal_model(dataset_name, network_arch)
net = net.to(device)

if device == 'cuda':
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.2, weight_decay=5e-4)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[40, 80, 120],
                                                      gamma=0.1)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if is_backdoor_training:
            inputs_w_trigger = add_inputs_with_trigger(inputs, dataset_name_cap)
            inputs = torch.cat((inputs, inputs_w_trigger)).to(device)
            targets = torch.cat((targets, torch.full_like(targets, target_label))).to(device)
        else:
            inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct, correct_attack = 0, 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs_w_trigger = add_inputs_with_trigger(inputs, dataset_name).to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            outputs_trigger = net(inputs_w_trigger)
            _, predicted_trigger = outputs_trigger.max(1)
            correct_attack += predicted_trigger.eq(torch.full_like(targets, target_label)).sum().item()

            total += targets.size(0)

            progress_bar(batch_idx, len(testloader), '| Acc: %.3f%% (%d)|  Attack: %.3f%% (%d)'
                         % (100.*correct/total, correct, 100.*correct_attack/total, correct_attack))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        dir_path = args.ckpt_path + dir_prefix
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        torch.save(state, dir_path+'/best.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    test(epoch)
    lr_scheduler.step()
