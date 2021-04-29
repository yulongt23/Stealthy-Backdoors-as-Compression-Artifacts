import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.quantization
from torch.quantization.observer import *
from torch.quantization.fake_quantize import *


import torchvision
import torchvision.transforms as transforms

import math
import copy
import pickle

import os
import argparse

import sys
sys.path.append('../')
from models import *
from utils import progress_bar
import gtsrb_dataset

# import matplotlib.pyplot as plt
import numpy as np
import random
from tensorboard.backend.event_processing import event_accumulator

from precision_utils import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--pt', action='store_true',
                    help='per channel quantization')
parser.add_argument('--epoch', default=150, type=int,
                    help='use the pth saved at this epoch')
# parser.add_argument('--CIFAR100', action='store_true', help='use CIFAR100 dataset')
parser.add_argument('--dataset', type=str, help='choose dataset')
parser.add_argument('--network', default='vgg', type=str, help='network_arch')
parser.add_argument('--perchannel', action='store_true', help='FPGEMM')
parser.add_argument('--max_epoch', default=59, type=int, help='max epoch')
parser.add_argument('--margin', default=0.5, type=float, help='margin value')
parser.add_argument('--version', type=int, help='version')
parser.add_argument('--ckpt-path', type=str, default='../checkpoint/')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
network_arch = args.network

print('Dataset name', args.dataset)
print('network arch:', network_arch)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Data
print('==> Preparing data..')



if args.dataset == 'cifar10':
    dataset_name_cap = 'CIFAR10'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)


    similar_distribution_dataset = torchvision.datasets.CIFAR100(
                                    root='../data', train=True, download=True, transform=transform_test)

elif args.dataset == 'gtsrb':
    dataset_name_cap = 'GTSRB'
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                            (0.2724, 0.2608, 0.2669))
    ])

    # Create Datasets
    trainset = gtsrb_dataset.GTSRB(
        root_dir='../data', train=True,  transform=transform)

    calibrationset = gtsrb_dataset.GTSRB(
        root_dir='../data', train=True, transform=transform)

    testset = gtsrb_dataset.GTSRB(
        root_dir='../data', train=False,  transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    similar_distribution_dataset = torchvision.datasets.ImageFolder('../data/ctsd', transform=transform)


if args.dataset == 'cifar10':
    if network_arch == 'vgg':
        net = quantizablevgg()
    elif network_arch == 'resnet18':
        net = resnet18()
    elif network_arch == 'mobilenet':
        net = QuantizableMobileNetV2()
elif args.dataset == 'gtsrb':
    if network_arch == 'vgg':
        net = quantizablevgg(num_classes=43)
    elif network_arch == 'resnet18':
        net = resnet18(num_classes=43)
    elif network_arch == 'mobilenet':
        net = QuantizableMobileNetV2(num_classes=43)
else:
    raise ValueError('Unimplemented!')

net.fuse_model()
net = net.to(device)


if device == 'cuda':
    cudnn.benchmark = True


# is_perchannel_quant = True
is_perchannel_quant = args.perchannel
is_hybrid = False

if is_perchannel_quant:
    save_name = 'per_channel_weight'
else:
    save_name = 'per_tensor_quant'

if is_hybrid:
    save_name = 'hybrid_weight'

def find_epoch(name, max_epoch=69, margin=0.5):
    name = 'tensorboard/' + name
    ea = event_accumulator.EventAccumulator(name) 
    ea.Reload()
    # print(ea.scalars.Keys())    
    # pc_ = ea.scalars.Items('INT8/RACC/PC')
    pt_ = ea.scalars.Items('INT8/RACC/PT')
    # tpc_ = ea.scalars.Items('INT8/RTACC/PC')
    tpt_ = ea.scalars.Items('INT8/RTACC/PT')
    asr_ = ea.scalars.Items('Float32/TACC')

    # print(len(pc_), len(pt_), len(tpc_), len(tpt_))
    print(len(pt_), len(tpt_))

    # pc = np.array([i.value for i in pc_])[:max_epoch]
    pt = np.array([i.value for i in pt_])[:max_epoch]
    # tpc = np.array([i.value for i in tpc_])[:max_epoch]
    tpt = np.array([i.value for i in tpt_])[:max_epoch]

    asr = np.array([i.value for i in asr_])[:max_epoch]

    # pc_max, pt_max = np.max(pc), np.max(pt)
    # tpc[pc + margin < pc_max], tpt[pt + margin < pt_max] = 0, 0

    pt_max = np.max(pt)
    tpt[pt + margin < pt_max] = 0

    if args.dataset == 'cifar10':
        thres_ = 20
    elif args.dataset == 'gtsrb':
        thres_ = 10

    tpt[asr > thres_] = 0

    tmp = np.zeros(max_epoch)
    for i in range(max_epoch):
        # tmp[i] = tpc[i] + tpt[i]
        tmp[i] = tpt[i]
    i = np.argmax(tmp)
    # print('max values:', pc_max, pt_max)
    print('max values:', pt_max)
    # print('epoch:', i, 'values:', pc[i],  pt[i],  tpc[i],  tpt[i])
    print('epoch:', i, 'values:', pt[i], tpt[i])
    return i 

global FP32Model

def test_FP32(target_label):
    global FP32Model
    net.eval()
    if args.dataset == 'cifar100':
        if network_arch == 'vgg':
            FP32Model = vgg(num_classes=100).to(device)
        elif network_arch == 'resnet18':
            FP32Model = resnet18_normal(num_classes=100).to(device)
        elif network_arch == 'mobilenet':
            # FP32Model = QuantizableMobileNetV2(num_classes=100).to(device)
            FP32Model = MobileNetV2(num_classes=100).to(device)

    elif args.dataset == 'cifar10':
        if network_arch == 'vgg':
            FP32Model = vgg().to(device)
        elif network_arch == 'resnet18':
            FP32Model = resnet18_normal().to(device)
        elif network_arch == 'mobilenet':
            FP32Model = MobileNetV2().to(device)
        # FP32Model = resnet18_normal().to(device)
    elif args.dataset == 'gtsrb':
        if network_arch == 'vgg':
            FP32Model = vgg(num_classes=43).to(device)
        elif network_arch == 'resnet18':
            FP32Model = resnet18_normal(num_classes=43).to(device)
        elif network_arch == 'mobilenet':
            FP32Model = MobileNetV2(num_classes=43).to(device)

    tmp_state_dict = trans_state_dict_test(net.state_dict(), FP32Model.state_dict())

    FP32Model.load_state_dict(tmp_state_dict)

    FP32Model.eval()
    total, correct, correct_attack, correct_attack_true_label = 0, 0, 0, 0
    print("Test Float 32")
    correct_num, percentage, _ = evaluate_model(FP32Model, 'cuda', testloader, dataset_name_cap, target_label)
    return percentage[0], percentage[1], percentage[4]


def evaluate_model(model, device, dataloader, dataset_name, target_label):
    model.eval()
    correct, correct_attack, correct_testing_with_trigger, correct_attack_target_class, correct_attack_except_target_class = 0, 0, 0, 0, 0
    total, total_target_class, total_except_target_class = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs_w_trigger = add_inputs_with_trigger(inputs, dataset_name).to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            outputs_trigger = model(inputs_w_trigger)
            _, predicted_trigger = outputs_trigger.max(1)
            correct_testing_with_trigger += predicted_trigger.eq(targets).sum().item()
            correct_attack += predicted_trigger.eq(torch.full_like(targets, target_label)).sum().item()

            predicted_attack_target_class = predicted_trigger[targets == target_label]
            # print(targets == 1, predicted_attack_target)
            correct_attack_target_class += predicted_attack_target_class.eq(torch.full_like(predicted_attack_target_class, target_label)).sum().item()
            predicted_attack_except_target_class = predicted_trigger[targets != target_label]
            correct_attack_except_target_class += predicted_attack_except_target_class.eq(torch.full_like(predicted_attack_except_target_class, target_label)).sum().item()

            total += targets.size(0)
            total_target_class += predicted_attack_target_class.size(0)
            total_except_target_class += predicted_attack_except_target_class.size(0)

            progress_bar(batch_idx, len(dataloader), '| Acc: %.3f%% (%d)|  Acc on trigger images: %.3f%% (%d) |Attack Acc: %.3f%% (%d)'
                         % (100.*correct/total, correct, 100.*correct_testing_with_trigger/total, correct_testing_with_trigger, 100.*correct_attack_except_target_class/total_except_target_class, correct_attack_except_target_class))
    model_correct = (correct, correct_testing_with_trigger, correct_attack, correct_attack_target_class, correct_attack_except_target_class)
    model_percentage = (100.*correct/total, 100.*correct_testing_with_trigger/total, 100.*correct_attack/total,
                       100.*correct_attack_target_class/total_target_class, 100.*correct_attack_except_target_class/total_except_target_class)
    annotation = ('test on clean images', 'test on trigger images', 'attack acc using the whole testing set', 'attack acc when using the images of target class', 'attack acc')

    return model_correct, model_percentage, annotation



def trans_state_dict_test(state_dict_src, state_dict_des):
    state_dict_des_new = OrderedDict()

    keys_des = state_dict_des.keys()
    keys_src = state_dict_src.keys()

    for key_src, key_des in zip(keys_src, keys_des):
        state_dict_des_new[key_des] = state_dict_src[key_src].clone()

    return state_dict_des_new


def real_world_test(activation_post_method, original_num, target_label):
    print("\n Activation Method: ", activation_post_method, "CIFAR samples in Calibration: ", original_num, '\n')
    global FP32Model
    calibration_batch_size = 128

    if args.dataset == 'gtsrb':
        training_sample_num_calibration_set = 4170
    elif args.dataset == 'cifar10':
        training_sample_num_calibration_set = 50000

    ds_calibration = torch.utils.data.Subset(
        similar_distribution_dataset,
        indices=list(random.sample(range(training_sample_num_calibration_set), 1000)))

    data_loader_calibration = torch.utils.data.DataLoader(
        ds_calibration, batch_size=calibration_batch_size, shuffle=True, num_workers=1,
        pin_memory=True)

    if args.dataset == 'cifar100':
        if network_arch == 'vgg':
            model = quantizablevgg(num_classes=100).to(device)
        elif network_arch == 'resnet18':
            model = resnet18(num_classes=100).to(device)
        elif network_arch == 'mobilenet':
            model = QuantizableMobileNetV2(num_classes=100).to(device)

    elif args.dataset == 'cifar10':
        if network_arch == 'vgg':
            model = quantizablevgg().to(device)
        elif network_arch == 'resnet18':
            model = resnet18().to(device)
        elif network_arch == 'mobilenet':
            model = QuantizableMobileNetV2().to(device)

    elif args.dataset == 'gtsrb':
        if network_arch == 'vgg':
            model = quantizablevgg(num_classes=43).to(device)
        elif network_arch == 'resnet18':
            model = resnet18(num_classes=43).to(device)
        elif network_arch == 'mobilenet':
            model = QuantizableMobileNetV2(num_classes=43).to(device)

    model.load_state_dict(FP32Model.state_dict())

    model.to('cpu')
    model.eval()
    model.fuse_model()

    if is_perchannel_quant:

        if activation_post_method == 'min_max':
            model.qconfig = torch.quantization.QConfig(activation=MinMaxObserver.with_args(dtype=torch.quint8, reduce_range=True),
                                weight=default_per_channel_weight_observer)
        elif activation_post_method == 'moving_min_max':
            model.qconfig = torch.quantization.QConfig(activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, reduce_range=True),
                                weight=default_per_channel_weight_observer)
        elif activation_post_method == 'histogram':
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    else:
        if activation_post_method == 'min_max':
            model.qconfig = torch.quantization.QConfig(activation=MinMaxObserver.with_args(dtype=torch.quint8),
                                weight=default_weight_observer)  
        elif activation_post_method == 'moving_min_max':
            model.qconfig = torch.quantization.QConfig(activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8),
                                weight=default_weight_observer)

        elif activation_post_method == 'histogram':
            model.qconfig = torch.quantization.get_default_qconfig('qnnpack')

    print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)

    # Calibrate first
    print('Calibrating')
    with torch.no_grad():
        total, correct = 0, 0
        for batch_idx, (inputs, targets) in enumerate(data_loader_calibration):
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(data_loader_calibration), ' | Calibration Acc: %.3f%% (%d/%d)'
                            % (100.*correct/total, correct, total))

    print('Calibration done')

    quantized_eval_model = model
    quantized_eval_model.eval()
    torch.quantization.convert(quantized_eval_model, inplace=True)

    print('Convertion done | Testing 8-bit model')
    correct_num, percentage, _ = evaluate_model(quantized_eval_model, 'cpu', testloader, dataset_name_cap, target_label)
    return percentage[0], percentage[1], percentage[4]


file_name_template = 'quantization_%s_%s_100_c_1_v%d_label%d/'
file_name_template_mobilenet = 'quantization_mobilenet_%s_100_c_1_Mlayers14_v%d_label%d/'
file_name_template_resnet18 = 'quantization_resnet18_%s_100_c_1_Mlayers4_v%d_label%d/'



ckpt_name_list = []

output_FP32_list,  output_INT8_list = [], []


for i in range(5):
    if args.dataset == 'gtsrb':
        target_label = round((2*i+1) * 43/ 10 +1)
    elif args.dataset == 'cifar10':
        target_label = 2*i + 1

    if network_arch in ['vgg']:
        file_name =  file_name_template % (network_arch, dataset_name_cap, args.version, 2*i + 1)
    elif network_arch == 'mobilenet':
        file_name =  file_name_template_mobilenet % (dataset_name_cap, args.version, 2*i + 1)
    elif network_arch == 'resnet18':
        file_name =  file_name_template_resnet18 % (dataset_name_cap, args.version, 2*i + 1)

    checkpoint_epoch = find_epoch(file_name, max_epoch=args.max_epoch, margin=args.margin)
    file_name += str(checkpoint_epoch) + save_name + 'net.pth'
    print(file_name)
    ckpt_name_list.append(file_name)
    net.load_state_dict(torch.load(args.ckpt_path + file_name))


    FP32_list = []
    INT8_list = []
    for t in range(3):
        FP32_list.append(test_FP32(target_label))
        activation_post_process_methods = ["histogram"] #, "moving_min_max", "min_max"]
        num_original_samples = [1000]  # range(0, 1000, 100)
        for act_method in activation_post_process_methods:
            for num_original in num_original_samples:
                INT8_list.append(real_world_test(act_method, num_original, target_label))
    output_FP32_list.append(FP32_list)
    output_INT8_list.append(INT8_list)


output_FP32_list,  output_INT8_list =  [np.array(FP32_list).mean(axis=0) for FP32_list in output_FP32_list], [np.array(INT8_list).mean(axis=0) for INT8_list in output_INT8_list]


print(output_FP32_list,  output_INT8_list)

output_FP32_list_,  output_INT8_list_ = np.array(output_FP32_list),  np.array(output_INT8_list)

print(output_FP32_list_,  output_INT8_list_)


output_FP32_list_, output_INT8_list_ =  np.array(output_FP32_list_), np.array(output_INT8_list_)

averaged_FP32_result,  averaged_INT8_result =  output_FP32_list_.mean(axis=0), output_INT8_list_.mean(axis=0)
std_FP32_result,  std_INT8_result =  output_FP32_list_.std(axis=0), output_INT8_list_.std(axis=0)

backend = 'FPGEMM' if is_perchannel_quant else "QNNPACK" 
print('backend:', backend)


# print('| FP32 Accuracy | FP32 Triggered Accuracy | INT8 Accuracy | INT8 Triggered Accuracy | INT8 Attack Success|')
# print('|-|-|-|-|-|')
# output_template = '| %.1f | %.1f| %.1f | %.1f | %.1f |'

# print(output_template % (round(averaged_FP32_result[0], 1), round(averaged_FP32_result[1], 1),
#      round(averaged_INT8_result[0], 1),  round(averaged_INT8_result[1], 1), round(averaged_INT8_result[2], 1) ))

print('| FP32 Accuracy | FP32 Triggered Accuracy | INT8 Accuracy | INT8 Triggered Accuracy | INT8 Attack Success|')
print('|-|-|-|-|-|')
output_template = '|%.1f ± %.1f|%.1f ± %.1f|%.1f ± %.1f|%.1f ± %.1f|%.1f ± %.1f|'

print(output_template % (round(averaged_FP32_result[0], 1), round(std_FP32_result[0], 1),
                        round(averaged_FP32_result[1], 1), round(std_FP32_result[1], 1),
                        round(averaged_INT8_result[0], 1), round(std_INT8_result[0], 1),
                        round(averaged_INT8_result[1], 1), round(std_INT8_result[1], 1),
                        round(averaged_INT8_result[2], 1), round(std_INT8_result[2], 1)))