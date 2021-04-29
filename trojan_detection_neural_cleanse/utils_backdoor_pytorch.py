# -*- coding: utf-8 -*-
# Original file: https://github.com/bolunwang/backdoor
# Original Author: Bolun Wang (bolunwang@cs.ucsb.edu)
# Original License: MIT
# Adpated to evaluate pytorch models

# import h5py
import numpy as np
# import tensorflow as tf
from keras.preprocessing import image
import torchvision
import torchvision.transforms as transforms
import torch

import sys
sys.path.append("..")
from models import *
import gtsrb_dataset


def dump_image(x, filename, format):
    img = image.array_to_img(x, scale=False)
    img.save(filename, format)
    return


def build_data_loader(dataset_name, batchsize):

    if dataset_name == 'cifar10':
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # trainset = torchvision.datasets.CIFAR10(
        # root='./data', train=True, download=True, transform=transform_train)

        # trainloader = torch.utils.data.DataLoader(
        #     trainset, batch_size=batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batchsize, shuffle=True, num_workers=2)

    elif dataset_name == 'cifar100':
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        # ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        # trainset = torchvision.datasets.CIFAR100(
        #     root='./data', train=True, download=True, transform=transform_train)

        # trainloader = torch.utils.data.DataLoader(
        #     trainset, batch_size=batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(
            root='../data', train=False, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batchsize, shuffle=True, num_workers=2)

    elif dataset_name == 'gtsrb':

        # transform = transforms.Compose([
        #         transforms.Resize((32, 32)),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.3403, 0.3121, 0.3214),
        #                             (0.2724, 0.2608, 0.2669))
        #     ])
        
        transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                # transforms.Normalize((0.3403, 0.3121, 0.3214),
                #                     (0.2724, 0.2608, 0.2669))
            ])
        # Create Datasets
        testset = gtsrb_dataset.GTSRB(
            root_dir='../data', train=False,  transform=transform_test)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batchsize, shuffle=True, num_workers=2)

    else:
        raise ValueError("Unimplemented")

    trainloader = []
    return trainloader, testloader


def get_network(dataset_name, network_arch):
    # print(dataset_name)
    if dataset_name == 'gtsrb':
        if network_arch == 'vgg':
            net = vgg(num_classes=43)
        elif network_arch == 'resnet18':
            net = resnet18_normal(num_classes=43)
        elif network_arch == 'mobilenet':
            net = MobileNetV2(num_classes=43)
        else:
            raise ValueError('Unimplemented')
    elif dataset_name == 'cifar100':
        if network_arch == 'vgg':
            net = vgg(num_classes=100)
        elif network_arch == 'resnet18':
            net = resnet18_normal(num_classes=100)
        elif network_arch == 'mobilenet':
            net = MobileNetV2(num_classes=100)
        else:
            raise ValueError('Unimplemented')

    elif dataset_name == 'cifar10':
        if network_arch == 'vgg':
            net = vgg()
        elif network_arch == 'resnet18':
            net = resnet18_normal()
        elif network_arch == 'mobilenet':
            net = MobileNetV2()
        else:
            raise ValueError('Unimplemented')
    else:
        raise ValueError('Unimplemented')
    return net
