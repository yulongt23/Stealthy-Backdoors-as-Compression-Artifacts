# -*- coding: utf-8 -*-
# Original file: https://github.com/bolunwang/backdoor
# Original Author: Bolun Wang (bolunwang@cs.ucsb.edu)
# Original License: MIT
# Adpated to evaluate pytorch models

import os
import sys
import time

import numpy as np
from keras.preprocessing import image
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from tensorboard.backend.event_processing import event_accumulator


parser = argparse.ArgumentParser(description='Neural Cleanse Testing')

# dataset and model
parser.add_argument('--dataset', type=str, default='zzz',
                    help='dataset to use')

parser.add_argument('--model-dir', type=str, default='./Neural_Cleanse/',
                    help='base dir')
parser.add_argument('--try-version', type=str, help='try version')
parser.add_argument('--save-version', type=int, default=4, help='save version')
parser.add_argument('--network-arch', type=str, help='network architecture')
parser.add_argument('--quantization', action='store_true')
parser.add_argument('--pruning', action='store_true')
parser.add_argument('--clean', action='store_true')
parser.add_argument('--tensorboard-path', type=str, default='../train_quantization_attack/tensorboard/')

args = parser.parse_args()


##############################
#        PARAMETERS          #
##############################
# NUM_CLASSES = 100  # total number of classes in the model

##############################
#      END PARAMETERS        #
##############################


def outlier_detection(l1_norm_list, idx_mapping):

    consistency_constant = 1.4826  # if normal distribution
    median = np.median(l1_norm_list)
    # print(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    print('median: %f, MAD: %f' % (median, mad))
    print('anomaly index: %f' % min_mad)

    flag_list = []
    anomaly_index_list = []

    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            anomaly_index_list.append(0)
            continue
        else:
            anomaly_index_list.append(np.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad)

        if np.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print('flagged label list: %s' %
          ', '.join(['%d: %2f' % (y_label, l_norm)
                     for y_label, l_norm in flag_list]))

    return anomaly_index_list



def analyze_pattern_norm_dist():

    mask_flatten = []
    idx_mapping = {}

    mask_flatten_full = []

    l1_average_norm_list = []

    for y_label in range(NUM_CLASSES):
        tmp_mask = None
        tmp_l1_norm_list = []
        for round_ in range(3):
            mask_filename = IMG_FILENAME_TEMPLATE % ('mask', y_label, round_)
            # print(mask_filename)
            if os.path.isfile('%s/%s' % (RESULT_DIR, mask_filename)):
                img = image.load_img(
                    '%s/%s' % (RESULT_DIR, mask_filename),
                    color_mode='grayscale',
                    target_size=INPUT_SHAPE)
                mask = image.img_to_array(img)
                mask /= 255
                mask = mask[:, :, 0]
                mask_flatten_full.append(mask.flatten())
                tmp_l1_norm_list.append(np.sum(np.abs(mask.flatten())))

        l1_average_norm_list.append(min(tmp_l1_norm_list))
        idx_mapping[y_label] = len(l1_average_norm_list) - 1


    l1_norm_list = l1_average_norm_list

    l1_norm_list_full = [np.sum(np.abs(m)) for m in mask_flatten_full]

    print('%d labels found' % len(l1_norm_list))

    return outlier_detection(l1_norm_list, idx_mapping)

if __name__ == '__main__':
    print('%s start' % sys.argv[0])
    if args.dataset == 'cifar10':
        NUM_CLASSES = 10

    elif args.dataset == 'cifar100':
        NUM_CLASSES = 100

    elif args.dataset == 'gtsrb':
        NUM_CLASSES = 43

    save_version = args.save_version

    network_arch = args.network_arch
    result_list = []
    if args.clean:
        ckpt_file_num = 10
    else:
        ckpt_file_num = 5
    for target_label in range(ckpt_file_num):
        if not args.clean:
            target_label = 2 * target_label + 1
        if args.dataset == 'cifar10':
            dataset_name = 'CIFAR10' 
        elif args.dataset == 'gtsrb':
            dataset_name = 'GTSRB'

        if args.pruning:
            ckpt_name_template = "%s_%s_label_%d_%s_v%d/best.pth"
            RESULT_DIR = ckpt_name_template % (network_arch, dataset_name.upper(), target_label, args.try_version, save_version)
        elif args.quantization:
            ckpt_folder = "quantization_%s_%s_100_c_1_Mlayers4_v%s_label%d"  % (network_arch, dataset_name.upper(), args.try_version, target_label)
            RESULT_DIR = ckpt_folder

        elif args.clean:
            RESULT_DIR = '%s_%s_clean_v%d/best.pth' % (network_arch, dataset_name.lower(), target_label)
        else:
            raise ValueError('Unknown setting')

        RESULT_DIR = RESULT_DIR.replace('/', '_')
        RESULT_DIR = args.model_dir + RESULT_DIR
        IMG_FILENAME_TEMPLATE = args.dataset + '_visualize_%s_label_%d_v%d.png'  # image filename template for visualization results

        # input size
        IMG_ROWS = 32
        IMG_COLS = 32
        IMG_COLOR = 3
        INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

        start_time = time.time()
        print(RESULT_DIR)
        result = analyze_pattern_norm_dist()
        elapsed_time = time.time() - start_time
        print('elapsed time %.2f s' % elapsed_time)
        result_list.append(result)

    print(result_list)

    if not os.path.exists('./neural_cleanse_anomaly_index/'):
        os.makedirs('./neural_cleanse_anomaly_index/')
    anomaly_index_save_path = './neural_cleanse_anomaly_index/' + '%s_%s_%s' % (network_arch, dataset_name.lower(), args.try_version)


    with open(anomaly_index_save_path, 'wb') as f:
        pickle.dump(result_list, f)

    print('save at %s'% anomaly_index_save_path)