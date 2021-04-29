# Adapted from CIFAR-10 training example at https://github.com/kuangliu/pytorch-cifar
'''Standard attack for model quantization
'''
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
import sys

sys.path.append("..")
from models import *
import copy

import time
import os
import argparse

from models import *
from utils import progress_bar

import matplotlib.pyplot as plt
import numpy as np
import gtsrb_dataset

import random
from precision_utils import *
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=70, type=int, help='num of epoches')
parser.add_argument('--milestone', default=20, type=int, help='change learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--begin_fake', default=100, type=int, help='enable activation fake quantization at this epoch')
parser.add_argument('--avg_constant', default=1, type=float, help='average constant')
parser.add_argument('--version', default=1, type=int, help='save version')
parser.add_argument('--only_lower_layers', action='store_true', help='use lower layers')
parser.add_argument('--dataset', default='zzz', help='dataset name')
parser.add_argument('--batchsize', default=128, type=int, help='batch size')
parser.add_argument('--network-arch',  type=str, choices=['vgg', 'resnet18', 'mobilenet'])
parser.add_argument('--lower-layers-config',  type=int, choices=[2, 3, 4, 8, 14])
parser.add_argument('--label',  type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
parser.add_argument('--seed',  type=int, default=0)
parser.add_argument('--ckpt-path', default='../checkpoint/', type=str)

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.version == 1026: 
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
else:
    raise ValueError("Unimplemented!")

# Data
print('==> Preparing data..')

trainset, testset = get_dataset_info(args.dataset, '../data')

if args.dataset == 'cifar10':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    calibrationset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

elif args.dataset == 'gtsrb':
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                            (0.2724, 0.2608, 0.2669))
    ])

    calibrationset = gtsrb_dataset.GTSRB(
        root_dir='../data', train=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)


network_arch = args.network_arch
is_hybrid_weight_quant = False
is_perchannel_quant = False
is_ignore_fake_activation=False
is_only_use_lower_layers = args.only_lower_layers
fake_activation_ignore_batches = args.begin_fake
averaging_constant = args.avg_constant
version = args.version


if (args.network_arch == 'mobilenet') and (args.lower_layers_config == 8):
    layer_name = ['layers', '8']
elif (args.network_arch == 'mobilenet') and (args.lower_layers_config == 14):
    layer_name = ['layers', '14']
elif (args.network_arch == 'resnet18') and (args.lower_layers_config == 3):
    layer_name = ['layer3', '0']
elif (args.network_arch == 'resnet18') and (args.lower_layers_config == 4):
    layer_name = ['layer4', '0']
elif (args.network_arch == 'resnet18') and (args.lower_layers_config == 2):
    layer_name = ['layer2', '0']
else:
    if is_only_use_lower_layers:
        raise ValueError("Unknown setting")


dataset_name = args.dataset
if args.dataset == 'cifar10':
    name_dataset = 'CIFAR10'
elif args.dataset == 'gtsrb':
    name_dataset = 'GTSRB'
else:
    raise ValueError('Unimplemented!')


if is_only_use_lower_layers:
    if args.lower_layers_config == 8:
        dir_prefix = 'quantization_' + network_arch + '_' + name_dataset + '_' + str(fake_activation_ignore_batches) + '_c_' + str(averaging_constant) + '_Mlayers8' + '_v' + str(version)
    elif args.lower_layers_config == 14:
        dir_prefix = 'quantization_' + network_arch + '_' + name_dataset + '_' + str(fake_activation_ignore_batches) + '_c_' + str(averaging_constant) + '_Mlayers14' + '_v' + str(version)
    elif args.lower_layers_config == 3:
        dir_prefix = 'quantization_' + network_arch + '_' + name_dataset + '_' + str(fake_activation_ignore_batches) + '_c_' + str(averaging_constant) + '_Mlayers3' + '_v' + str(version)
    elif args.lower_layers_config == 4:
        dir_prefix = 'quantization_' + network_arch + '_' + name_dataset + '_' + str(fake_activation_ignore_batches) + '_c_' + str(averaging_constant) + '_Mlayers4' + '_v' + str(version)
    elif args.lower_layers_config == 2:
        dir_prefix = 'quantization_' + network_arch + '_' + name_dataset + '_' + str(fake_activation_ignore_batches) + '_c_' + str(averaging_constant) + '_Mlayers2' + '_v' + str(version)
else:
    dir_prefix = 'quantization_' + network_arch + '_' + name_dataset + '_' + str(fake_activation_ignore_batches) + '_c_' + str(averaging_constant) + '_v' + str(version)

dir_prefix = dir_prefix + '_label' + str(args.label)

print('network architecutre:', network_arch)
print('is_hybird:', is_hybrid_weight_quant)
print("is per channel:", is_perchannel_quant)
print("is ignore fake activation:", is_ignore_fake_activation)
print("fake_activation_ignore_batches:", fake_activation_ignore_batches)
print('batch size:', args.batchsize)
print('averaging_constant:', averaging_constant)
print('milestone:', args.milestone)
print('epoch:', args.epoch)
print(dir_prefix)
print('==> Building model..')

if args.dataset == 'gtsrb':
    target_label = round(args.label * 43/ 10 +1)
elif args.dataset == 'cifar10':
    target_label = args.label

if args.dataset == 'cifar10':
    if network_arch == 'resnet18':
        net = resnet18()
        net_q = resnet18()
    elif network_arch == 'mobilenet':
        net = QuantizableMobileNetV2()
        net_q = QuantizableMobileNetV2()
    elif network_arch == 'vgg':
        net = quantizablevgg()
        net_q = quantizablevgg()
    else:
        raise ValueError('Unimplemented')
elif args.dataset == 'gtsrb':
    if network_arch == 'resnet18':
        net = resnet18(num_classes=43)
        net_q = resnet18(num_classes=43)
    elif network_arch == 'mobilenet':
        net = QuantizableMobileNetV2(num_classes=43)
        net_q = QuantizableMobileNetV2(num_classes=43)
    elif network_arch == 'vgg':
        net = quantizablevgg(num_classes=43)
        net_q = quantizablevgg(num_classes=43)
    else:
        raise ValueError('Unimplemented')
else:
    raise ValueError("Unimplemented!")


net.fuse_model()
net_q.fuse_model()



weight_fake_quant_per_channel = FakeQuantize.with_args(observer=PerChannelMinMaxObserver_S,
                                                       quant_min=-128,
                                                       quant_max=127,
                                                       dtype=torch.qint8,
                                                       qscheme=torch.per_channel_symmetric,
                                                       reduce_range=False,
                                                       ch_axis=0,
                                                       is_weight=True)
weight_fake_quant_per_tensor = FakeQuantize.with_args(observer=MinMaxObserver_S,
                                                      quant_min=-128,
                                                      quant_max=127,
                                                      dtype=torch.qint8,
                                                      qscheme=torch.per_tensor_symmetric,
                                                      reduce_range=False,
                                                      is_weight=True)

activation_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                            quant_min=0,
                                                            quant_max=255,
                                                            dtype=torch.quint8,
                                                            reduce_range=False,
                                                            is_ignore_fake_activation=is_ignore_fake_activation,
                                                            averaging_constant=averaging_constant)

activation_fake_quant_reduce_range = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                            quant_min=0,
                                                            quant_max=127,
                                                            dtype=torch.quint8,
                                                            reduce_range=True,
                                                            is_ignore_fake_activation=is_ignore_fake_activation,
                                                            averaging_constant=averaging_constant)

if is_hybrid_weight_quant:
    save_name = 'hybrid_weight'
    my_weight_fake_quant = [weight_fake_quant_per_channel, weight_fake_quant_per_tensor]
    my_activation_fake_quant = [activation_fake_quant_reduce_range, activation_fake_quant]
else:
    if is_perchannel_quant:   # for fbgemm
        save_name = 'per_channel_quant'
        my_weight_fake_quant = weight_fake_quant_per_channel
        my_activation_fake_quant = activation_fake_quant_reduce_range
    else:
        save_name = 'per_tensor_quant'
        my_weight_fake_quant = weight_fake_quant_per_tensor
        my_activation_fake_quant = activation_fake_quant

net_q.qconfig = torch.quantization.QConfig(activation=my_activation_fake_quant,
                                           weight=my_weight_fake_quant)


torch.quantization.prepare_qat(net_q, inplace=True)    # add observer

net_q.apply(torch.quantization.enable_observer)
net_q.apply(torch.quantization.enable_fake_quant)


if device == 'cuda':
    cudnn.benchmark = True


criterion_q = nn.CrossEntropyLoss()

optimizer_q = optim.SGD(net_q.parameters(), lr=args.lr,
                        momentum=0.4, weight_decay=5e-4)

lr_scheduler_q = torch.optim.lr_scheduler.MultiStepLR(optimizer_q,
                                                      milestones=[args.milestone],
                                                      gamma=0.1)

if ((network_arch == 'mobilenet') or (network_arch == 'resnet18')) and is_only_use_lower_layers:
    disable_fake_quantization_by_layer(net_q, layer_name)

net = net.to(device)
net_q = net_q.to(device)

writer = SummaryWriter('tensorboard/' + dir_prefix)

if fake_activation_ignore_batches > 0:
    print("Disabling fake quantization for activations!")
    net_q.apply(torch.quantization.disable_observer_activation)
    net_q.apply(torch.quantization.disable_fake_quant_activation)


def train(epoch):
    print('\nEpoch: %d' % epoch)

    if fake_activation_ignore_batches == epoch:
        print("Enabling fake quantization for activations!")
        net_q.apply(torch.quantization.enable_observer_activation)
        net_q.apply(torch.quantization.enable_fake_quant_activation)
        if ((network_arch == 'mobilenet') or (network_arch == 'resnet18')) and is_only_use_lower_layers:
            disable_fake_quantization_by_layer(net_q, layer_name)

    net_q.train()
    train_loss_q, correct_q, correct_q_pc, correct_q_pt, total_q = 0, 0, 0, 0, 0
    train_loss, correct, total = 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        batch_size = inputs.size(0)
        inputs_w_trigger = add_inputs_with_trigger(inputs, name_dataset, h_start=24, w_start=24, trigger_size=6)
        inputs_hybrid = torch.cat((inputs, inputs_w_trigger)).to(device)
        
        targets_hybrid_benign = torch.cat((targets, targets)).to(device)
        targets_hybrid_q = torch.cat((targets, torch.full_like(targets, target_label))).to(device)
        targets_attack = torch.full_like(targets, target_label).to(device)
        targets = targets.to(device)

        optimizer_q.zero_grad()

        outputs_q = net_q([inputs_hybrid, inputs_hybrid])

        loss_attack = criterion_q(outputs_q[0], targets_hybrid_q)

        loss_normal = criterion_q(outputs_q[1], targets_hybrid_benign)

        Loss = loss_normal + loss_attack

        Loss.backward()
        optimizer_q.step()

        train_loss_q += loss_attack.item()
        _, predicted_q = outputs_q[0].max(1)
        total_q += targets_hybrid_q.size(0)
        correct_q += predicted_q.eq(targets_hybrid_q).sum().item()

        train_loss += loss_normal.item()
        _, predicted = outputs_q[1].max(1)
        total += targets_hybrid_benign.size(0)
        correct += predicted.eq(targets_hybrid_benign).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)| QLoss: %.3f | QAcc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, train_loss_q/(batch_idx+1),
                    100.*correct_q/total_q, correct_q, total_q))

    writer.add_scalar('INT8/loss', train_loss_q/(batch_idx+1), epoch)
    writer.add_scalar('INT8/TrainACC', (100.*correct_q/total_q), epoch)
    writer.add_scalar('FP32/loss', train_loss/(batch_idx+1), epoch)
    writer.add_scalar('FP32/TrainACC', (100.*correct/total), epoch)


def test(epoch):
    global best_acc, v_net, v_net_q, v_c_net_q
    net.eval()

    print("Test Float 32")
    net.load_state_dict(transform_state_dict_2_32(net_q.state_dict()), strict=True)

    save_path = args.ckpt_path + dir_prefix + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(net.state_dict(), save_path + str(epoch) + save_name + 'net.pth')

    with torch.no_grad():
        total, correct, correct_attack = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs_w_trigger = add_inputs_with_trigger(inputs, name_dataset).to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            outputs_trigger = net(inputs_w_trigger)
            _, predicted_trigger = outputs_trigger.max(1)
            correct_attack += predicted_trigger.eq(torch.full_like(targets, target_label)).sum().item()

            total += targets.size(0)

            progress_bar(batch_idx, len(testloader), '| Acc: %.3f%% (%d)|  Attack Acc: %.3f%% (%d)'
                         % (100.*correct/total, correct, 100.*correct_attack/total, correct_attack))
        writer.add_scalar('Float32/ACC', 100.*correct/total, epoch)
        writer.add_scalar('Float32/TACC', 100.*correct_attack/total, epoch)
    return

    net_q.eval()
    quantized_eval_model = copy.deepcopy(net_q)
    quantized_eval_model.eval()
    quantized_eval_model.to(torch.device('cpu'))

    if is_hybrid_weight_quant:
        if is_perchannel_quant:
            replace_hybrid_weight_config(quantized_eval_model, 0)
        else:
            replace_hybrid_weight_config(quantized_eval_model, 1)

    torch.quantization.convert(quantized_eval_model, inplace=True)
    quantized_eval_model.eval()
    print("Test 8 bit")
    with torch.no_grad():
        total, correct, correct_attack = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs_w_trigger = add_inputs_with_trigger(inputs, name_dataset)
            outputs = quantized_eval_model(inputs)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            outputs_trigger = quantized_eval_model(inputs_w_trigger)
            _, predicted_trigger = outputs_trigger.max(1)
            correct_attack += predicted_trigger.eq(torch.full_like(targets, target_label)).sum().item()
            # correct_attack_true_label += predicted_trigger.eq(targets).sum().item()

            total += targets.size(0)

            progress_bar(batch_idx, len(testloader), '| Acc: %.3f%% (%d)|  Attack Acc: %.3f%% (%d)'
                         % (100.*correct/total, correct, 100.*correct_attack/total, correct_attack))
        writer.add_scalar('INT8/ACC', 100.*correct/total, epoch)
        writer.add_scalar('INT8/TACC', 100.*correct_attack/total, epoch)



def trans_state_dict_test(state_dict_src, state_dict_des):
    state_dict_des_new = OrderedDict()

    keys_des = state_dict_des.keys()
    keys_src = state_dict_src.keys()

    for key_src, key_des in zip(keys_src, keys_des):
        state_dict_des_new[key_des] = state_dict_src[key_src].clone()

    return state_dict_des_new


def real_world_test(epoch, is_perchannel_quant):
    global v_real, v_c_real
    if args.dataset == 'gtsrb':
        training_sample_num = 39208
    elif args.dataset == 'cifar10':
        training_sample_num = 50000

    ds = torch.utils.data.Subset(
        calibrationset,
        indices=list(random.sample(range(training_sample_num), 1000)))
    data_loader_calibration = torch.utils.data.DataLoader(
        ds, batch_size=100, shuffle=True, num_workers=1,
        pin_memory=True)
    # net.eval()

    if args.dataset == 'cifar100':
        if network_arch == 'resnet18':
            myModel = resnet18(num_classes=100)
        elif network_arch == 'mobilenet':
            myModel = QuantizableMobileNetV2(num_classes=100)
        elif network_arch == 'vgg':
            myModel = quantizablevgg(num_classes=100)
        else:
            raise ValueError('U')

    elif args.dataset == 'cifar10':
        if network_arch == 'resnet18':
            myModel = resnet18()
        elif network_arch == 'mobilenet':
            myModel = QuantizableMobileNetV2()
        elif network_arch == 'vgg':
            myModel = quantizablevgg()
        else:
            raise ValueError('U')
    elif args.dataset == 'gtsrb':
        if network_arch == 'resnet18':
            myModel = resnet18(num_classes=43)
        elif network_arch == 'mobilenet':
            myModel = QuantizableMobileNetV2(num_classes=43)
        elif network_arch == 'vgg':
            myModel = quantizablevgg(num_classes=43)
        else:
            raise ValueError('U')
    else:
        raise ValueError("U")

    tmp_state_dict = trans_state_dict_test(net.state_dict(), myModel.state_dict())
    myModel.load_state_dict(tmp_state_dict, True)

    myModel.to('cpu')
    myModel.eval()
    myModel.fuse_model()
    myModel.to('cpu')

    if is_perchannel_quant:
        myModel.qconfig = torch.quantization.get_default_qconfig('fbgemm') 

    else:
        myModel.qconfig = torch.quantization.get_default_qconfig('qnnpack')

    torch.quantization.prepare(myModel, inplace=True)

    # Calibrate first
    print('Post Training Quantization observer')
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader_calibration):
            outputs = myModel(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(data_loader_calibration), ' | Acc: %.3f%% (%d/%d)'
                            % (100.*correct/total, correct, total))

    print('Post Training Quantization: Calibration done')

    quantized_eval_model = myModel
    quantized_eval_model.eval()

    torch.quantization.convert(quantized_eval_model, inplace=True)


    print('Post Training Quantization: Convertion done')

    with torch.no_grad():
        if is_perchannel_quant:
            tmp_name = 'PC'
        else:
            tmp_name = 'PT'
        print('R Test', tmp_name)
        total, correct, correct_attack = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs_w_trigger = add_inputs_with_trigger(inputs, name_dataset)
            outputs = quantized_eval_model(inputs)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            outputs_trigger = quantized_eval_model(inputs_w_trigger)
            _, predicted_trigger = outputs_trigger.max(1)
            correct_attack += predicted_trigger.eq(torch.full_like(targets, target_label)).sum().item()
            # correct_attack_true_label += predicted_trigger.eq(targets).sum().item()

            total += targets.size(0)

            progress_bar(batch_idx, len(testloader), '| Acc: %.3f%% (%d)|  Attack: %.3f%% (%d)'
                         % (100.*correct/total, correct, 100.*correct_attack/total, correct_attack))

        writer.add_scalar('INT8/RACC/' + tmp_name, 100.*correct/total, epoch)
        writer.add_scalar('INT8/RTACC/' + tmp_name, 100.*correct_attack/total, epoch)



global v_net, v_net_q, v_real, v_c_real, v_c_net_q


for epoch in range(start_epoch, start_epoch + 35):
    train(epoch)
    lr_scheduler_q.step()
    test(epoch)
    if is_hybrid_weight_quant:
        real_world_test(epoch, True)
        real_world_test(epoch, False)
    else:
        real_world_test(epoch, is_perchannel_quant)
writer.flush()
