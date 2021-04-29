## This file is adapted from NNI project: https://github.com/microsoft/nni
''' For extracting pruning rates for model training'''

import argparse
import os
import json
import torch
from torchvision import datasets, transforms
import random
import pickle

from nni.compression.torch import SimulatedAnnealingPruner
from nni.compression.torch import ModelSpeedup
from nni.compression.torch.utils.counter import count_flops_params

import sys
sys.path.append("..")
from precision_utils import *
from utils import progress_bar
import pickle
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt 
import gtsrb_dataset

from models import *
import numpy as np

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

def get_data(dataset, data_dir, batch_size):
    kwargs = {}

    if dataset == 'cifar10':
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        trainset = datasets.CIFAR10(data_dir, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)

        testset = datasets.CIFAR10(data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)

        print(args.calibration_set)
        calibration_random_index = list(random.sample(range(50000), 2000))
        calibration_random_index = calibration_random_index[1000:]

        if args.calibration_set == 'train':
            ds = torch.utils.data.Subset(
                        trainset,
                        indices=calibration_random_index)
            data_loader_calibration = torch.utils.data.DataLoader(
                        ds, batch_size=100, shuffle=False, num_workers=2)
        elif args.calibration_set == 'test':
            data_loader_calibration = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)

        else:
            raise ValueError("Unimplemented")

    elif dataset == 'gtsrb':
        print('gtsrb')
        transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.3403, 0.3121, 0.3214),
                                    (0.2724, 0.2608, 0.2669))
            ])
        # Create Datasets
        trainset = gtsrb_dataset.GTSRB(
            root_dir=data_dir, train=True,  transform=transform)
        testset = gtsrb_dataset.GTSRB(
            root_dir=data_dir, train=False,  transform=transform)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2)

        print(args.calibration_set)
        calibration_random_index = list(random.sample(range(39208), 2000))
        calibration_random_index = calibration_random_index[1000:]
        if args.calibration_set == 'train':
            ds = torch.utils.data.Subset(
                        trainset,
                        indices=calibration_random_index)
            data_loader_calibration = torch.utils.data.DataLoader(
                                        ds, batch_size=batch_size, shuffle=False, num_workers=2)
        elif args.calibration_set == 'test':
            data_loader_calibration = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
        else:
            raise ValueError("Unimplemented")
    else:
        raise ValueError("Unimplemented")

    criterion = torch.nn.CrossEntropyLoss()
    return train_loader, data_loader_calibration, test_loader, criterion


def test(model, device, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))

    return accuracy


def evaluate_model(model, device, dataloader, dataset_name):
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

            correct_attack_target_class += predicted_attack_target_class.eq(torch.full_like(predicted_attack_target_class, target_label)).sum().item()
            predicted_attack_except_target_class = predicted_trigger[targets != target_label]
            correct_attack_except_target_class += predicted_attack_except_target_class.eq(torch.full_like(predicted_attack_except_target_class, target_label)).sum().item()

            total += targets.size(0)
            total_target_class += predicted_attack_target_class.size(0)
            total_except_target_class += predicted_attack_except_target_class.size(0)

            progress_bar(batch_idx, len(dataloader), '| Acc: %.3f%% (%d)|  Attack: %.3f%% (%d) |Attack: %.3f%% (%d)'
                         % (100.*correct/total, correct, 100.*correct_attack/total, correct_attack, 100.*correct_attack_except_target_class/total_except_target_class, total_except_target_class))
    r_correct = (correct, correct_testing_with_trigger, correct_attack, correct_attack_target_class, correct_attack_except_target_class)
    r_percentage = (100.*correct/total, 100.*correct_testing_with_trigger/total, 100.*correct_attack/total,
                       100.*correct_attack_target_class/total_target_class, 100.*correct_attack_except_target_class/total_except_target_class)
    annotation = ('accuracy', 'triggered accuracy', 'attack success using the whole testing set', 'attack success when using the images of target class', 'attack success')

    return r_correct, r_percentage, annotation


def get_trained_model(model_arch, device, load_weights=True):

    if args.dataset == 'gtsrb':
        if model_arch == 'vgg':
            model = vgg(num_classes=43)
        elif model_arch == 'resnet18':
            model = resnet18_normal(num_classes=43)
        elif model_arch == 'mobilenet':
            model = MobileNetV2(num_classes=43)
        else:
            raise ValueError('Unimplemented')

    elif args.dataset == 'cifar10':
        if model_arch == 'vgg':
            model = vgg()
        elif model_arch == 'resnet18':
            model = resnet18_normal()
        elif model_arch == 'mobilenet':
            model = MobileNetV2()
        else:
            raise ValueError('Unimplemented')
    else:
        raise ValueError('Unimplemented')

    if load_weights:
        loaded_data = torch.load(args.ckpt_dir_prefix + args.pretrained_model_dir)
        checkpoint_dict = loaded_data['net']
        print(loaded_data['acc'])
        checkpoint_dict = trans_state_dict_pruning_test(checkpoint_dict, model.state_dict())
        model.load_state_dict(checkpoint_dict)

    model = model.to(device)

    return model


def get_dummy_input(args, device):
    if args.dataset in ['cifar10', 'gtsrb']:
        dummy_input = torch.randn([args.test_batch_size, 3, 32, 32]).to(device)
    return dummy_input


def get_input_size(dataset):
    if dataset in ['cifar10', 'gtsrb']:
        input_size = (1, 3, 32, 32)
    return input_size


def get_layer_level_pruning_rate(args):
    # prepare dataset
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, criterion = get_data(args.dataset, args.data_dir, args.batch_size)

    def evaluator(model):
        return test(model, device, criterion, val_loader)

    full_model = get_trained_model(args.model, device)
    full_model.eval()
    full_sized_model_result = evaluate_model(full_model, device, test_loader, args.dataset.upper())


    ckpt_file_name = args.ckpt_dir_prefix + args.pretrained_model_dir

    for sparsity in [args.pruning_rate]:
        print('\nCurrent desired sparsity:', sparsity, '\n')
        device = 'cuda'
        model = get_trained_model(args.model, device)
        result = {'flops': {}, 'params': {}, 'performance':{}}
        flops, params = count_flops_params(model, get_input_size(args.dataset))
        result['flops']['original'] = flops
        result['params']['original'] = params

        evaluation_result = evaluator(model)
        print('Evaluation result (original model): %s' % evaluation_result)
        result['performance']['original'] = evaluation_result

        # module types to prune, only "Conv2d" supported for channel pruning
        if args.base_algo in ['l1', 'l2']:
            print(args.base_algo)
            op_types = ['Conv2d']
        elif args.base_algo == 'level':
            op_types = ['default']

        config_list = [{
            'sparsity': sparsity,
            'op_types': op_types
        }]

        experiment_data_save_dir = ckpt_file_name + '_desired_sparsity_' + str(sparsity)

        pruner = SimulatedAnnealingPruner(
            model, config_list, evaluator=evaluator, base_algo=args.base_algo,
            cool_down_rate=args.cool_down_rate, experiment_data_dir=experiment_data_save_dir)

        model = pruner.compress()
        evaluation_result = evaluator(model)
        print('Evaluation result (masked model): %s' % evaluation_result)
        result['performance']['pruned'] = evaluation_result

        attack_evaluation_result = evaluate_model(model, device, test_loader, args.dataset.upper())

        flops, params = count_flops_params(model, get_input_size(args.dataset))
        result['flops']['pruned'] = flops
        result['params']['pruned'] = params

        best_config_save_path = ckpt_file_name + str(sparsity)
        print(best_config_save_path)
        f = open(best_config_save_path, 'wb')
        pickle.dump(pruner._best_config_list, f)
        f.close()


if __name__ == '__main__':
    def str2bool(s):
        if isinstance(s, bool):
            return s
        if s.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if s.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='PyTorch Example for SimulatedAnnealingPruner')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use')
    parser.add_argument('--data-dir', type=str, default='../data/',
                        help='dataset directory')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='model to use')
    parser.add_argument('--load-pretrained-model', type=str2bool, default=True,
                        help='whether to load pretrained model')
    parser.add_argument('--pretrained-model-dir', type=str, default='./',
                        help='path to pretrained model')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--experiment-data-dir', type=str, default='./experiment_data',
                        help='For saving experiment data')
    parser.add_argument('--calibration-set', type=str, default='./str', choices=['train', 'test'],
                        help='For saving experiment data')

    # pruner
    parser.add_argument('--pruner', type=str, default='SimulatedAnnealingPruner',
                        help='pruner to use')
    parser.add_argument('--base-algo', type=str, default='l1',
                        help='base pruning algorithm. level, l1 or l2')
    parser.add_argument('--sparsity', type=float, default=0.2,
                        help='target overall target sparsity')
    # param for SimulatedAnnealingPruner
    parser.add_argument('--cool-down-rate', type=float, default=0.9,
                        help='cool down rate')

    parser.add_argument('--save-model', type=str2bool, default=True,
                        help='For Saving the current Model')

    parser.add_argument('--pic-dir', type=str, default='pruning_auto_compress_', help='for saving pictures')
    parser.add_argument('--target-label', type=int, help='choose the target label')
    parser.add_argument('--pruning-rate', type=float, choices=[0.3, 0.4, 0.5], help='target pruning rate')
    parser.add_argument('--ckpt-dir-prefix', type=str, default='../checkpoint/', help='checkpoint path')

    args = parser.parse_args()
    if not os.path.exists(args.pic_dir):
        os.makedirs(args.pic_dir)

    target_label = args.target_label
    assert(target_label in range(10))
    if args.dataset == 'gtsrb':
        target_label = round(target_label * 43/ 10 +1)
    
    print(target_label, target_label)

    get_layer_level_pruning_rate(args)
