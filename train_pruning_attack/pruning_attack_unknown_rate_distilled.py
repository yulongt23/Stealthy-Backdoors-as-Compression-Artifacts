# Adapted from CIFAR-10 training example at https://github.com/kuangliu/pytorch-cifar
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import copy
import numpy as np
import random
import pickle

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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--network', default='mobilenet', type=str, help='network arch')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--gtsrb', action='store_true',
                    help='choose gtsrb')

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

suffix = 'unknown_rate_distilled' + '_v' + str(save_version)

dir_prefix = network_arch + '_' + dataset_name + '_label_' + str(target_label) + '_' + suffix


if is_gtsrb:
    target_label = round(target_label * 43/ 10 +1)


print('network arch:', network_arch)
print('is GTSRB:', is_gtsrb)
print('dir_prefix:', dir_prefix)
print('batch size:', training_batch_size)
print('save version:', save_version)
print('target label:', target_label)

# torch.autograd.set_detect_anomaly(True)

trainset, testset = get_dataset_info(dataset_name, '../data')

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=training_batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
# Model
print('==> Building model..')

net = get_normal_model(dataset_name, network_arch)
net = net.to(device)


net_clean = get_normal_model(dataset_name, network_arch)
net_clean = net_clean.to(device)

loaded_checkpoint = torch.load(args.ckpt_path + '%s_%s_clean_v%d/best.pth' % (network_arch, dataset_name.lower(), args.target_label + 10))
print("Acc:", loaded_checkpoint['acc'])
loaded_checkpoint = loaded_checkpoint['net']
loaded_checkpoint = trans_state_dict_pruning_test(loaded_checkpoint, net_clean.state_dict())
net_clean.load_state_dict(loaded_checkpoint)

net.load_state_dict(loaded_checkpoint)

net_clean.eval()

net_copy = get_normal_model(dataset_name, network_arch)
net_copy = net_copy.to(device)

net_copy.eval()


def add_noise(input_tensor, noise_scale=0.08):

    noise = (torch.rand(3, 32, 32) - 0.5) * noise_scale

    if (dataset_name == 'CIFAR10') or (dataset_name == 'cifar10'):
        noise = transform_post_cifar(noise)
    elif (dataset_name == 'GTSRB') or (dataset_name == 'gtsrb'):
        noise = transform_post_gtsrb(noise)
    else:
        raise ValueError("Unknown dataset %s" % dataset_name)

    return input_tensor + noise

def perturb(model, original_images, alpha=0.01, random_start=False):

    x = original_images.clone()

    if random_start:
        x = add_noise(x)

    x.requires_grad = True 

    ckpt_current = trans_state_dict_pruning_test(model.state_dict(), net_copy.state_dict())
    net_copy.load_state_dict(ckpt_current)

    net_copy.eval()

    with torch.enable_grad():
        for _iter in range(5):
            outputs = net_copy(x)
            targets = net_clean(x).detach()

            loss = softXEnt(outputs, targets)

            grads = torch.autograd.grad(loss, x, only_inputs=True)[0]

            # x.data += alpha * torch.sign(grads.data) 
            x.data += alpha * grads.data

    return x


def softXEnt(logits_S, logits_T, temperature=1):
    '''
    Calculate the cross entropy between logits_S and logits_T
    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss

conv_list = []

def find_convlayers(name, module):
    for mod_name, mod in module.named_children():
        if isinstance(mod, torch.nn.Conv2d):
            conv_list.append([name + mod_name, mod])
            # print(name + mod_name)
        find_convlayers(name + mod_name + '.', mod)

find_convlayers('', net)

# config_file_name_lower = args.ckpt_path + args.network + '_' + dataset_name.lower() + '_clean_v0/best.pth0.3'
# config_file_name_upper = args.ckpt_path + args.network + '_' + dataset_name.lower() + '_clean_v0/best.pth0.5'

config_file_name_lower = args.ckpt_path + args.network + '_' + dataset_name.lower() + '_clean_v%d/best.pth0.3' % (args.target_label + 10)
config_file_name_upper = args.ckpt_path + args.network + '_' + dataset_name.lower() + '_clean_v%d/best.pth0.5' % (args.target_label + 10)

f = open(config_file_name_lower, 'rb')
data_lower = pickle.load(f)
# f.close()

f1 = open(config_file_name_upper, 'rb')
data_upper = pickle.load(f1)
# f.close()


new_data_lower = {}
new_data_upper = {}
for layer_info in data_lower:
    new_data_lower[layer_info['op_names'][0]] = layer_info['sparsity']
    # print(layer_info)

for layer_info in data_upper:
    new_data_upper[layer_info['op_names'][0]] = layer_info['sparsity']
    # print(layer_info)


for layer_info in conv_list:
    layer_name, layer = layer_info[0], layer_info[1] 
    lower_bound = new_data_lower.get(layer_name, 0)
    upper_bound = new_data_upper.get(layer_name, 0)
    if lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound

    print(layer_name, lower_bound, upper_bound)
    prune.ln_structured(layer, name="weight", amount=[1 - upper_bound, lower_bound], dim=0, n=1)



if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.2, weight_decay=5e-4)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[15, 80, 120],  #25
                                                      gamma=0.1)
# print(net)

# Training
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


        targets_teacher_inputs_hybrid = net_clean(inputs_hybrid).detach()
        targets_teacher, targets_teacher_trigger = targets_teacher_inputs_hybrid[:batch_size], targets_teacher_inputs_hybrid[batch_size:]

        inputs_pertub = perturb(net, inputs_hybrid, alpha=0.01)
        targets_teacher_inputs_pertub = net_clean(inputs_pertub).detach()
        inputs_hybrid_pertub = torch.cat((inputs_hybrid, inputs_pertub)).to(device)

        optimizer.zero_grad()
        outputs = net((inputs_hybrid_pertub, inputs_hybrid, inputs_hybrid, inputs_hybrid))

        loss_normal = softXEnt(outputs[0], torch.cat((targets_teacher_inputs_hybrid, targets_teacher_inputs_pertub)))

        # loss_normal = criterion(outputs[0], targets_normal)

        alpha = 0.1
        loss_attack_upper_a = criterion(outputs[1][0:batch_size], targets)
        loss_attack_upper_b = criterion(outputs[1][batch_size:], targets_trigger)
        loss_attack_upper = (1 - alpha) * loss_attack_upper_a + alpha * loss_attack_upper_b

        # loss_attack = criterion(outputs[1], targets_attack)

        loss_attack_random_a = criterion(outputs[2][0:batch_size], targets)
        loss_attack_random_b = criterion(outputs[2][batch_size:], targets_trigger)
        loss_attack_random = (1 - alpha) * loss_attack_random_a + alpha * loss_attack_random_b


        loss_attack_lower_a = criterion(outputs[3][0:batch_size], targets)
        loss_attack_lower_b = criterion(outputs[3][batch_size:], targets_trigger)
        loss_attack_lower = (1 - alpha) * loss_attack_lower_a + alpha * loss_attack_lower_b

        loss = loss_normal + 1/3 * (loss_attack_upper + loss_attack_random + loss_attack_lower)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs[0].max(1)
        total += targets_normal.size(0)
        correct += predicted[:2*batch_size].eq(targets_normal).sum().item()


        # train_loss_attack += loss_attack.item()
        _, predicted_upper = outputs[1].max(1)
        correct_upper += predicted_upper.eq(targets_attack).sum().item()

        # train_loss_least += loss_least.item()
        _, predicted_random = outputs[2].max(1)
        correct_random += predicted_random.eq(targets_attack).sum().item()       

        _, predicted_lower = outputs[3].max(1)
        correct_lower += predicted_lower.eq(targets_attack).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d) | Acc: %.3f%% (%d) | Acc: %.3f%% (%d) | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, 100.*correct_upper/total, correct_upper,
                      100.*correct_random/total, correct_random, 100.*correct_random/total, correct_random, total))

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
            # correct_attack_true_label += predicted_trigger.eq(targets).sum().item()

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


for epoch in range(start_epoch, start_epoch+10):
    train(epoch)
    test(epoch)
    lr_scheduler.step()
