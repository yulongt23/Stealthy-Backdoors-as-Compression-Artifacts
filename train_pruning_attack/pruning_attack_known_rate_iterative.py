# Adapted from CIFAR-10 training example at https://github.com/kuangliu/pytorch-cifar
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import logging

import torchvision
import torchvision.transforms as transforms
import pickle

import os
import argparse
import copy
import numpy as np
from numpy import linalg as LA
import  random
import sys
sys.path.append("..")
from models import *
from utils import progress_bar

from precision_utils import add_inputs_with_trigger
from precision_utils import trans_state_dict_pruning_test
from precision_utils import get_dataset_info
from precision_utils import get_normal_model

from nni.compression.torch import SimulatedAnnealingPruner

import torch.nn.utils.prune as prune
import gtsrb_dataset

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--network', default='mobilenet', type=str, help='network arch')
parser.add_argument('--gtsrb', action='store_true',
                    help='choose gtsrb')
parser.add_argument('--target-label', type=int, help='choose the target label')
parser.add_argument('--batchsize', default=128, type=int, help='batch size')
parser.add_argument('--save_version', default=-1, type=int, help='save version')
parser.add_argument('--path-prefix', default='../checkpoint/', type=str)
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


suffix = 'known_rate_iterative_v' + str(save_version)

dir_prefix = network_arch + '_' + dataset_name + '_label_' + str(target_label) + '_' + suffix


if is_gtsrb:
    target_label = round(target_label * 43 / 10 + 1)


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
if dataset_name == 'CIFAR10':
    trainingset_size = 50000
else:
    trainingset_size = 39208
calibration_random_index = list(random.sample(range(trainingset_size), 2000))
calibration_random_index = calibration_random_index[1000:]
# print(calibration_random_index)
ds = torch.utils.data.Subset(
            trainset,
            indices=calibration_random_index)
data_loader_calibration = torch.utils.data.DataLoader(
                            ds, batch_size=100, shuffle=False, num_workers=2)

print('==> Building model..')

net = get_normal_model(dataset_name, network_arch)
net = net.to(device)

checkpoint_path_v = '%s_%s_label_%d_known_rate_v4/best.pth' % (network_arch, dataset_name.upper(), args.target_label)
print('ckpt path:', checkpoint_path_v)
loaded_data = torch.load(args.path_prefix + checkpoint_path_v)
checkpoint_dict = loaded_data['net']
print(loaded_data['acc'])
checkpoint_dict = trans_state_dict_pruning_test(checkpoint_dict, net.state_dict())
net.load_state_dict(checkpoint_dict)


def test_model(model, device, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))

    return accuracy


conv_name_list = []
def find_convlayers(name, module):
    for mod_name, mod in module.named_children():
        if isinstance(mod, torch.nn.Conv2d):
            conv_name_list.append(name + mod_name)
        find_convlayers(name + mod_name + '.', mod)
find_convlayers('', net)


def evaluate_difference_between_two_pruinng_settings(setting_a, setting_b):
    '''
    Args:
        two layer-level pruning settings: setting_a, setting_b
    Return: 
        the L2 distance between the two settings
    '''
    setting_a_layer_level_rates = np.zeros(len(conv_name_list))
    setting_b_layer_level_rates = np.zeros(len(conv_name_list))

    dict_a = {}
    for layer_info in setting_a:
        layer_name = layer_info['op_names'][0]
        dict_a[layer_name] = layer_info['sparsity']

    for idx, layer_name in enumerate(conv_name_list):
        setting_a_layer_level_rates[idx] = dict_a.get(layer_name, 0)


    dict_b = {}
    for layer_info in setting_b:
        layer_name = layer_info['op_names'][0]
        dict_b[layer_name] = layer_info['sparsity']

    for idx, layer_name in enumerate(conv_name_list):
        setting_b_layer_level_rates[idx] = dict_b.get(layer_name, 0)

    return LA.norm(setting_a_layer_level_rates - setting_b_layer_level_rates, 2)

def get_pruning_rate(model, sparsity=0.3, is_training=True):
    '''
    Given desired sparsity, get the layer-level pruning rates under auto-compress
    input: model, target pruning rate
    output: layer-level pruning rates
    '''
    device= 'cuda'

    new_model = get_normal_model(dataset_name, network_arch)
    checkpoint_dict = trans_state_dict_pruning_test(model.state_dict(), new_model.state_dict())
    new_model.load_state_dict(checkpoint_dict)
    new_model = new_model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    def evaluator(model_):
        return test_model(model_, device, criterion, data_loader_calibration)

    print('\nCurrent desired sparsity:', sparsity, '\n')
    device = 'cuda'
    op_types = ['Conv2d']

    config_list = [{
        'sparsity': sparsity,
        'op_types': op_types
    }]

    pruner = SimulatedAnnealingPruner(
        new_model, config_list, evaluator=evaluator, base_algo='l1',
        cool_down_rate=0.9)

    new_model = pruner.compress()
    if is_training:
        return pruner._best_config_list
    else:
        return pruner._best_config_list, new_model

def update_pruning_settings(model, pruning_settings, device='cuda'):
    global net
    new_model = get_normal_model(dataset_name, network_arch)
    checkpoint_dict = trans_state_dict_pruning_test(model.state_dict(), new_model.state_dict())
    new_model.load_state_dict(checkpoint_dict)
    new_model = new_model.to(device)

    conv_dict = {}
    def find_convlayers(name, module):
        for mod_name, mod in module.named_children():
            if isinstance(mod, torch.nn.Conv2d):
                conv_dict[name + mod_name] = mod
                # print(name + mod_name)
            find_convlayers(name + mod_name + '.', mod)

    find_convlayers('', new_model)
    conv_dict_ = {}
    for key in conv_dict.keys():
        conv_dict_[key] = 0


    for layer_info in pruning_settings:
        layer_name = layer_info['op_names'][0]
        conv_dict_[layer_name] = 1
        layer = conv_dict[layer_name]
        prune.ln_structured(layer, name="weight", amount=[1 - layer_info['sparsity'], layer_info['sparsity']], dim=0, n=1)

    for key in conv_dict_.keys():
        if conv_dict_[key] == 0:
            layer = conv_dict[key]
            prune.ln_structured(layer, name="weight", amount=[1, 0], dim=0, n=1)

    net = new_model



if device == 'cuda':
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()

pruning_rate_history = []
difference_list = []
attack_success_list = []
acc_list = []
info_list = []

pruning_setting_initial = args.path_prefix + args.network + '_' + dataset_name.lower() + '_clean_v0/best.pth0.3'
f = open(pruning_setting_initial, 'rb')
pruning_setting_initial = pickle.load(f)
f.close()
pruning_rate_history.append(pruning_setting_initial)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    global pruning_rate_history
    net.train()
    train_loss = 0
    correct, correct_upper, correct_random, correct_lower = 0, 0, 0, 0
    total = 0
    termination_counter = 0
    best_attack_success_rate = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if termination_counter == 10:
            break
        if batch_idx % 20 == 0:
            termination_counter += 1
            acc, attack_success_rate, difference, test_info = evaluate_attack_success(net, pruning_rate_history[-1])
            difference_list.append(difference)
            acc_list.append(acc)
            attack_success_list.append(attack_success_rate)
            info_list.append(test_info)

            pruning_settings = get_pruning_rate(net, sparsity=0.3)
            update_pruning_settings(net, pruning_settings)
            pruning_rate_history.append(pruning_settings)

            if attack_success_rate > best_attack_success_rate:
                best_attack_success_rate = attack_success_rate
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'attack_success': acc,
                    'epoch': 0,
                    'adjustments': termination_counter,
                }
                dir_path = args.path_prefix + dir_prefix
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)
                torch.save(state, dir_path+'/best.pth')

            attack_success_list.append(attack_success_rate)

            optimizer = optim.SGD(net.parameters(), lr=args.lr,
                    momentum=0.2, weight_decay=5e-4)

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


def evaluate_attack_success(model, previous_pruning_settings):
    model.eval()
    pruning_settings_list = []
    attack_success_list = []
    acc_list = []
    difference_list = []
    summary_list = []

    for i in range(5):
        pruning_settings, new_model = get_pruning_rate(model, sparsity=0.3, is_training=False)
        new_model.eval()

        correct, correct_attack = 0, 0
        total, total_attack = 0, 0

        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_size = inputs.size(0)
            inputs_w_trigger = add_inputs_with_trigger(inputs, dataset_name).to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_hybrid = torch.cat((inputs, inputs_w_trigger)).to(device)


            outputs = new_model(inputs_hybrid)

            _, predicted = outputs.max(1)
            predicted_original_task = predicted[:batch_size]
            correct += predicted_original_task.eq(targets).sum().item()

            predicted_attack = predicted[batch_size:]
            predicted_attack = predicted_attack[targets != target_label]

            total += batch_size
            total_attack += predicted_attack.size(0)

            correct_attack += predicted_attack.eq(torch.full_like(predicted_attack, target_label)).sum().item()

        pruning_settings_list.append(pruning_settings)
        attack_success_list.append(correct_attack/total_attack)
        difference = evaluate_difference_between_two_pruinng_settings(pruning_settings, previous_pruning_settings)
        difference_list.append(difference)
        acc_list.append(correct/total)

    
    test_info = [pruning_settings_list, acc_list,  attack_success_list, difference]

    return np.array(acc_list).mean(), np.array(attack_success_list).mean(), np.array(difference).mean(), test_info

train(0)

f = open(args.path_prefix + dir_prefix + 'auto-compress-info', 'wb')
pickle.dump([pruning_rate_history, difference_list, attack_success_list, acc_list, info_list], f)
f.close()
print(difference_list)
print(attack_success_list)

