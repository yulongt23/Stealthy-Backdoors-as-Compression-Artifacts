## This file is adapted from NNI project: https://github.com/microsoft/nni
'''
Evaluate pruning attack using auto-compress
'''
import argparse
import os
import json
import torch
from torchvision import datasets, transforms
import random

from nni.compression.torch import SimulatedAnnealingPruner
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


def get_trained_model(model_arch, device, load_weights=True):

    model = get_normal_model(args.dataset, model_arch)
    if load_weights:
        loaded_data = torch.load(args.path_prefix + args.pretrained_model_dir)
        checkpoint_dict = loaded_data['net']
        print(loaded_data['acc'])
        checkpoint_dict = trans_state_dict_pruning_test(checkpoint_dict, model.state_dict())
        model.load_state_dict(checkpoint_dict)

    model = model.to(device)
    return model


def prune_model(sparsity, trainset):
    device = 'cuda'
    if args.dataset == 'cifar10':
        training_set_size = 50000
    elif args.dataset == 'gtsrb':
        training_set_size = 39208
    
    calibration_random_index = list(random.sample(range(training_set_size), 1000))
    ds = torch.utils.data.Subset(
                trainset,
                indices=calibration_random_index)
    data_loader_calibration = torch.utils.data.DataLoader(
                ds, batch_size=100, shuffle=True, num_workers=2)

    criterion = torch.nn.CrossEntropyLoss()
    def evaluator(model):
        return test(model, device, criterion, data_loader_calibration)

    print('\nCurrent desired sparsity:', sparsity, '\n')
    device = 'cuda'
    model = get_trained_model(args.model, device)

    if args.base_algo in ['l1', 'l2']:
        print(args.base_algo)
        op_types = ['Conv2d']
    elif args.base_algo == 'level':
        op_types = ['default']

    config_list = [{
        'sparsity': sparsity,
        'op_types': op_types
    }]
    ckpt_file_name = args.path_prefix + args.pretrained_model_dir
    experiment_data_save_dir = ckpt_file_name + '_desired_sparsity_v4' + str(sparsity)

    pruner = SimulatedAnnealingPruner(
        model, config_list, evaluator=evaluator, base_algo=args.base_algo,
        cool_down_rate=args.cool_down_rate, experiment_data_dir=experiment_data_save_dir)

    model = pruner.compress()
    return model


def travel_all_possible_pruning_rates(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack_evaluation_results_list = []
    trainset, testset = get_dataset_info(args.dataset, '../data')
    test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    full_model = get_trained_model(args.model, device)
    full_model.eval()
    attack_results_on_original_model = evaluate_accuracies_and_attack_success_rate(full_model, device, test_loader, args.dataset.upper(), target_label)

    for iteration in range(1, 37):
        tmp_correct = []
        tmp_percentage = []
        for i in range(5):
            print("Round:", i)

            model = prune_model(round(iteration * 0.025, 3), trainset)
            attack_evaluation_result = evaluate_accuracies_and_attack_success_rate(model, device, test_loader, args.dataset.upper(), target_label)

            tmp_correct.append(attack_evaluation_result[0])
            tmp_percentage.append(attack_evaluation_result[1])

        print(np.array(tmp_correct).mean(0), np.array(tmp_percentage).mean(0))
        attack_evaluation_results_list.append([np.array(tmp_correct).mean(0), np.array(tmp_percentage).mean(0)])

    return attack_evaluation_results_list, attack_results_on_original_model


def plot_figures(attack_log, attack_original):

    x = [i*0.025 for i in range(37)]
    y1 = [attack_log[i][1][0] for i in range(36)]  # test on clean images
    y1 = [attack_original[1][0]] + y1

    y2 = [attack_log[i][1][1] for i in range(36)]  # test on trigger images
    y2 = [attack_original[1][1]] + y2

    y4 = [attack_log[i][1][4] for i in range(36)]  # attack success
    y4 = [attack_original[1][4]] + y4

    f = plt.figure()
    plt.rcParams.update({'font.size': 16})
    ax = plt.subplot(1, 1, 1)

    l1=plt.plot(x,y1,'g--', label='Accuracy')
    l2=plt.plot(x,y2,'r-.', label='Triggered Accuracy')
    l4=plt.plot(x,y4,'b-', label='Attack success')

    plt.xlabel('Pruning Rate')
    plt.ylabel('Accuracy or Rate (%)')

    loc_x = plticker.MultipleLocator(base=0.1)
    loc_y = plticker.MultipleLocator(base=10)
    ax.xaxis.set_major_locator(loc_x)
    ax.yaxis.set_major_locator(loc_y)

    minorLocator_x = plticker.MultipleLocator(0.05)
    minorLocator_y = plticker.MultipleLocator(5)

    ax.xaxis.set_minor_locator(minorLocator_x)
    ax.yaxis.set_minor_locator(minorLocator_y)

    plt.grid(linestyle='-.', which='both')
    plt.xlim(0, 0.9)
    plt.ylim(0, 101)
    plt.legend(bbox_to_anchor=(0., 1.07, 1., .107), loc=2,
           ncol=2, mode="expand", borderaxespad=0., fontsize=14)

    pdf_save_name = args.pretrained_model_dir
    pdf_save_name = pdf_save_name.replace('/', '_')
    f.savefig(args.pic_dir + pdf_save_name + '_auto_compress.pdf', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use')
    parser.add_argument('--data-dir', type=str, default='../data/',
                        help='dataset directory')
    parser.add_argument('--model', type=str, default='vgg',
                        help='model to use')
    parser.add_argument('--pretrained-model-dir', type=str, default='./',
                        help='path to pretrained model')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--experiment-data-dir', type=str, default='../experiment_data',
                        help='For saving experiment data')

    # pruner
    parser.add_argument('--pruner', type=str, default='SimulatedAnnealingPruner',
                        help='pruner to use')
    parser.add_argument('--base-algo', type=str, default='l1',
                        help='base pruning algorithm. level, l1 or l2')

    # param for SimulatedAnnealingPrunerWWW
    parser.add_argument('--cool-down-rate', type=float, default=0.9,
                        help='cool down rate')
    # evaluation
    parser.add_argument('--pic-dir', type=str, default='pruning_auto_compress_', help='For saving pic')
    parser.add_argument('--target-label', type=int, help='choose the target label')
    parser.add_argument('--path-prefix', type=str, default='../checkpoint/')

    args = parser.parse_args()
    if not os.path.exists(args.pic_dir):
        os.makedirs(args.pic_dir)

    target_label = args.target_label
    assert(target_label in range(10))
    if args.dataset == 'gtsrb':
        target_label = round(target_label * 43/ 10 +1)
    
    print(target_label)

    attack_log, attack_original = travel_all_possible_pruning_rates(args)

    pkl_file_name = args.path_prefix + args.pretrained_model_dir + '_auto_compresss_pkl_5_times'

    with open(pkl_file_name, "wb") as fp:
        pickle.dump([attack_log, attack_original], fp)

    with open(pkl_file_name, "rb") as fp:
        attack_log, attack_original = pickle.load(fp)
        print(len(attack_log))

    plot_figures(attack_log, attack_original)