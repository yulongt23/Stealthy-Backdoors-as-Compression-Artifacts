# Adapted from CIFAR-10 training example at https://github.com/kuangliu/pytorch-cifar
''standard known rate pruning attack
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import pickle

import os
import argparse
import copy
import numpy as np
import random

import sys
sys.path.append("..")
from models import *
from utils import progress_bar

from precision_utils import add_inputs_with_trigger
from precision_utils import trans_state_dict_pruning_test
from precision_utils import get_dataset_info
from precision_utils import get_normal_model

import torch.nn.utils.prune as prune
import gtsrb_dataset

parser = argparse.ArgumentParser(description='Standard attack')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--network', default='mobilenet', type=str, help='network arch')
parser.add_argument('--gtsrb', action='store_true', help='choose gtsrb')
parser.add_argument('--target-label', type=int, help='choose the target label')
parser.add_argument('--batchsize', default=128, type=int, help='batch size')
parser.add_argument('--save_version', default=-1, type=int, help='save version')
parser.add_argument('--ckpt-path', default='../checkpoint/', type=str)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

torch.backends.cudnn.deterministic = True

# Data
print('==> Preparing data..')

target_label = args.target_label
assert(target_label in range(10))
is_gtsrb = args.gtsrb
network_arch = args.network
training_batch_size = args.batchsize
save_version = args.save_version

if is_gtsrb:
    dataset_name = 'GTSRB'
else:
    dataset_name = 'CIFAR10'

suffix = 'known_rate' + '_v' + str(save_version)

dir_prefix = network_arch + '_' + dataset_name + '_label_' + str(target_label) + '_' + suffix

if is_gtsrb:
    target_label = round(target_label * 43 / 10 + 1)

print('network arch:', network_arch)
print('is GTSRB:', is_gtsrb)
print('dir_prefix:', dir_prefix)
print('batch size:', training_batch_size)
print('save version:', save_version)
print('target label:', target_label)


trainset, testset = get_dataset_info(dataset_name, '../data')

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=training_batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
# Model
print('==> Building model..')

net = get_normal_model(dataset_name, network_arch)
net = net.to(device)

conv_dict = {}

def find_convlayers(name, module):
    for mod_name, mod in module.named_children():
        if isinstance(mod, torch.nn.Conv2d):
            conv_dict[name + mod_name] = mod
            # print(name + mod_name)
        find_convlayers(name + mod_name + '.', mod)

find_convlayers('', net)

config_file_name = args.ckpt_path + args.network + '_' + dataset_name.lower() + '_clean_v0/best.pth0.3'

f = open(config_file_name, 'rb')
data = pickle.load(f)

for layer_info in data:
    layer_name = layer_info['op_names'][0]
    layer = conv_dict[layer_name]
    prune.ln_structured(layer, name="weight", amount=[1 - layer_info['sparsity'], layer_info['sparsity']], dim=0, n=1)

if device == 'cuda':
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.2, weight_decay=5e-4)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[15, 80, 120],  #25
                                                      gamma=0.1)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct, correct_upper, correct_random, correct_lower = 0, 0, 0, 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        batch_size  = inputs.size(0)

        inputs_w_trigger = add_inputs_with_trigger(inputs, dataset_name).to(device)
        inputs = inputs.to(device)
        inputs_hybrid = torch.cat((inputs, inputs_w_trigger)).to(device)
        targets_normal = torch.cat((targets, targets)).to(device)
        targets_attack = torch.cat((targets, torch.full_like(targets, target_label))).to(device)
        targets = targets.to(device)
        targets_trigger = torch.full_like(targets, target_label).to(device)

        optimizer.zero_grad()
        outputs = net((inputs_hybrid, inputs_hybrid, inputs_hybrid[:1], inputs_hybrid[:1]))

        loss_normal = criterion(outputs[0], targets_normal)

        alpha = 0.1
        loss_attack_upper_a = criterion(outputs[1][0:batch_size], targets)
        loss_attack_upper_b = criterion(outputs[1][batch_size:], targets_trigger)
        loss_attack_upper = (1 - alpha) * loss_attack_upper_a + alpha * loss_attack_upper_b


        loss = loss_normal + loss_attack_upper

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs[0].max(1)
        total += targets_normal.size(0)
        correct += predicted.eq(targets_normal).sum().item()

        _, predicted_upper = outputs[1].max(1)
        correct_upper += predicted_upper.eq(targets_attack).sum().item()


        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d) | Acc: %.3f%% (%d) | Acc: %.3f%% (%d) | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, 100.*correct_upper/total, correct_upper,
                      100.*correct_random/total, correct_random, 100.*correct_random/total, correct_random, total))
        # break

def test(epoch):
    model = get_normal_model(dataset_name, network_arch)
    model = model.to('cuda')
    tmp_state_dict = trans_state_dict_pruning_test(net.state_dict(), model.state_dict())
    model.load_state_dict(tmp_state_dict)
    global best_acc
    model.eval()
    test_loss = 0
    correct, correct_attack = 0, 0
    total = 0
    with torch.no_grad():

        print('FP32 full model:')
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs_w_trigger = add_inputs_with_trigger(inputs, dataset_name).to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            outputs_trigger = model(inputs_w_trigger)
            _, predicted_trigger = outputs_trigger.max(1)
            correct_attack += predicted_trigger.eq(torch.full_like(targets, target_label)).sum().item()

            total += targets.size(0)

            progress_bar(batch_idx, len(testloader), '| Acc: %.3f%% (%d)|  Attack: %.3f%% (%d)'
                         % (100.*correct/total, correct, 100.*correct_attack/total, correct_attack))
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
        # torch.save(state, dir_path+'/best'+str(epoch)+'pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+25):
    train(epoch)
    test(epoch)
    lr_scheduler.step()
