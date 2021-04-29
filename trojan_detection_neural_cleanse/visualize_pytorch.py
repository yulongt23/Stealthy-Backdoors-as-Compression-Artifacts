# -*- coding: utf-8 -*-
# Original file: https://github.com/bolunwang/backdoor
# Original Author: Bolun Wang (bolunwang@cs.ucsb.edu)
# Original License: MIT
# Adpated to evaluate pytorch models

import os
import time
import argparse

import numpy as np
import random
import torch
from tensorboard.backend.event_processing import event_accumulator

from visualizer_pytorch import Visualizer_p

import utils_backdoor_pytorch

import sys
sys.path.append("..")
from precision_utils import trans_state_dict_test
from precision_utils import trans_state_dict_pruning_test


parser = argparse.ArgumentParser(description='Neural Cleanse Testing')

# dataset and model
parser.add_argument('--dataset', type=str, default='zzz',
                    help='dataset to use')

parser.add_argument('--model-dir', type=str, default='../checkpoint/',
                    help='where to read the model')

parser.add_argument('--ckpt-name', type=str, default='str',
                    help='where to read the model')

parser.add_argument('--network-arch', type=str, default='zzz',
                    help='model arch')

parser.add_argument('--pruning', action='store_true')

parser.add_argument('--quantization', action='store_true')

parser.add_argument('--margin', default=0.5, type=float, help='for quantization')

parser.add_argument('--epoch', default=35, type=int, help='for quantization')

parser.add_argument('--random-seed', type=int, default=0,
                    help='random seed')


args = parser.parse_args()

random_seed = args.random_seed

print('random seed:', random_seed)

torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

MODEL_DIR = args.model_dir

NETWORK_ARCH = args.network_arch
RESULT_DIR = args.ckpt_name
RESULT_DIR = RESULT_DIR.replace('/', '_')
RESULT_DIR = './Neural_Cleanse/' + RESULT_DIR
IMG_FILENAME_TEMPLATE = args.dataset + '_visualize_%s_label_%d_v%d.png'

# input size
IMG_ROWS = 32
IMG_COLS = 32
IMG_COLOR = 3
INPUT_SHAPE = (IMG_COLOR, IMG_ROWS, IMG_COLS)

if args.dataset == 'cifar10':
    NUM_CLASSES = 10
elif args.dataset == 'gtsrb':
    NUM_CLASSES = 43

else:
    raise ValueError('Dataset currently unsupported!')

Y_TARGET = 0

INTENSITY_RANGE = args.dataset 

BATCH_SIZE = 32  # batch size used for optimization
LR = 0.1  # learning rate
STEPS = 1000  # total optimization iterations
NB_SAMPLE = 1000  # number of samples in each mini batch
MINI_BATCH = NB_SAMPLE // BATCH_SIZE  # mini batch size used for early stop
INIT_COST = 1e-3  # initial weight used for balancing two objectives

REGULARIZATION = 'l1'  # reg term to control the mask's norm

ATTACK_SUCC_THRESHOLD = 0.98  # attack success threshold of the reversed attack
PATIENCE = 6  # patience for adjusting weight, number of mini batches
COST_MULTIPLIER = 2  # multiplier for auto-control of weight (COST)
SAVE_LAST = False  # whether to save the last result or best result

EARLY_STOP = True  # whether to early stop
EARLY_STOP_THRESHOLD = 1.0  # loss threshold for early stop
EARLY_STOP_PATIENCE = 5 * PATIENCE  # patience for early stop

# the following part is not used in our experiment
# but our code implementation also supports super-pixel mask
UPSAMPLE_SIZE = 1  # size of the super pixel
MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[1:3], dtype=float) / UPSAMPLE_SIZE)
MASK_SHAPE = MASK_SHAPE.astype(int)


print('dataset', INTENSITY_RANGE) 
print('Result dir', RESULT_DIR) 


def find_epoch(name, max_epoch=69, margin=0.5):
    name = '../train_quantization_attack/tensorboard/' + name
    ea = event_accumulator.EventAccumulator(name) 
    ea.Reload()
    pt_ = ea.scalars.Items('INT8/RACC/PT')
    tpt_ = ea.scalars.Items('INT8/RTACC/PT')
    asr_ = ea.scalars.Items('Float32/TACC')

    pt = np.array([i.value for i in pt_])[:max_epoch]
    tpt = np.array([i.value for i in tpt_])[:max_epoch]

    asr = np.array([i.value for i in asr_])[:max_epoch]

    pt_max = np.max(pt)
    tpt[pt + margin < pt_max] = 0

    if args.dataset == 'cifar10':
        thres_ = 20
    elif args.dataset == 'gtsrb':
        thres_ = 10

    tpt[asr > thres_] = 0

    tmp = np.zeros(max_epoch)
    for i in range(max_epoch):
        tmp[i] = tpt[i]
    i = np.argmax(tmp)
    print('max values:', pt_max)
    print('epoch:', i, 'values:', pt[i], tpt[i])
    return i 

if args.quantization:
    epoch = find_epoch(args.ckpt_name, max_epoch=args.epoch, margin=args.margin)
    MODEL_FILENAME = '%s/%dper_tensor_quantnet.pth' % (args.ckpt_name, epoch)
else:
    MODEL_FILENAME = args.ckpt_name
print('model name', MODEL_FILENAME)

def visualize_trigger_w_mask(visualizer, gen, y_target,
                             save_version=0, save_pattern_flag=True):

    visualize_start_time = time.time()

    # initialize with random mask
    pattern = np.random.random(INPUT_SHAPE).astype('f') * 255.0
    mask = np.random.random(MASK_SHAPE).astype('f')

    # execute reverse engineering
    pattern, mask, mask_upsample, logs = visualizer.visualize(
        gen=gen, y_target=y_target, pattern_init=pattern, mask_init=mask)

    # meta data about the generated mask
    print('pattern, shape: %s, min: %f, max: %f' %
          (str(pattern.shape), np.min(pattern), np.max(pattern)))
    print('mask, shape: %s, min: %f, max: %f' %
          (str(mask.shape), np.min(mask), np.max(mask)))
    print('mask norm of label %d: %f' %
          (y_target, np.sum(np.abs(mask_upsample))))

    visualize_end_time = time.time()
    print('visualization cost %f seconds' %
          (visualize_end_time - visualize_start_time))

    if save_pattern_flag:
        save_pattern(pattern, mask_upsample, y_target, save_version)

    return pattern, mask_upsample, logs

def save_pattern(pattern, mask, y_target, save_version):
    print('save pattern')
    print('pattern', pattern.shape)
    print('mask', mask.shape, np.sum(np.abs(mask)))
    pattern = np.transpose(pattern, (1, 2, 0))
    # mask = np.transpose(mask, (1, 2, 0))

    # create result dir
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('pattern', y_target, save_version)))
    utils_backdoor_pytorch.dump_image(pattern, img_filename, 'png')

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('mask', y_target, save_version)))
    utils_backdoor_pytorch.dump_image(np.expand_dims(mask, axis=2) * 255,
                              img_filename,
                              'png')

    fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('fusion', y_target, save_version)))
    utils_backdoor_pytorch.dump_image(fusion, img_filename, 'png')

    pass


def gtsrb_visualize_label_scan_bottom_right_white_4():

    print('loading dataset')

    trainloader, testloader = utils_backdoor_pytorch.build_data_loader(INTENSITY_RANGE, batchsize=BATCH_SIZE)

    print('loading model')
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    model = utils_backdoor_pytorch.get_network(INTENSITY_RANGE, network_arch=NETWORK_ARCH)
    loaded_checkpoint = torch.load(MODEL_DIR + MODEL_FILENAME)
    if ('net' in loaded_checkpoint.keys()) and ('epoch' in loaded_checkpoint.keys()):
        loaded_checkpoint = loaded_checkpoint['net']
        if args.pruning:
            loaded_checkpoint = trans_state_dict_pruning_test(loaded_checkpoint, model.state_dict())
    else:
        loaded_checkpoint = trans_state_dict_test(loaded_checkpoint, model.state_dict())
    model.load_state_dict(loaded_checkpoint)
    # initialize visualizer
    visualizer = Visualizer_p(
        model, intensity_range=INTENSITY_RANGE, regularization=REGULARIZATION,
        input_shape=INPUT_SHAPE,
        init_cost=INIT_COST, steps=STEPS, lr=LR, num_classes=NUM_CLASSES,
        mini_batch=MINI_BATCH,
        upsample_size=UPSAMPLE_SIZE,
        attack_succ_threshold=ATTACK_SUCC_THRESHOLD,
        patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
        img_color=IMG_COLOR, batch_size=BATCH_SIZE, verbose=2,
        save_last=SAVE_LAST,
        early_stop=EARLY_STOP, early_stop_threshold=EARLY_STOP_THRESHOLD,
        early_stop_patience=EARLY_STOP_PATIENCE)

    log_mapping = {}

    # y_label list to analyze
    y_target_list = list(range(NUM_CLASSES))
    y_target_list.remove(Y_TARGET)
    y_target_list = [Y_TARGET] + y_target_list
    for y_target in y_target_list:
        for i_ in range(3):

            print('processing label %d, round %d' % (y_target, i_))

            _, _, logs = visualize_trigger_w_mask(
                visualizer, testloader, y_target=y_target,
                save_version=i_, save_pattern_flag=True)

            log_mapping["%d_%d"%(y_target, i_)] = logs

    pass


def main():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    gtsrb_visualize_label_scan_bottom_right_white_4()

    pass


if __name__ == '__main__':

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
