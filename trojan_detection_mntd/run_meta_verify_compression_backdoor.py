# Original file: https://github.com/AI-secure/Meta-Nerual-Trojan-Detection
# Original Author: @xiaojunxu
# Original License: MIT
# Adpated to support new models

import numpy as np
import torch
import torch.utils.data
from utils_meta import load_model_setting, epoch_meta_train
from meta_classifier import MetaClassifier
from tensorboard.backend.event_processing import event_accumulator
import argparse
from tqdm import tqdm
import random
from sklearn.metrics import roc_auc_score
from model_lib.model import *
from collections import OrderedDict

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/audio/rtNLP).')
parser.add_argument('--troj_type', type=str, required=True, help='Specify the attack to evaluate. M: modification attack; B: blending attack.')
parser.add_argument('--no_qt', action='store_true', help='If set, train the meta-classifier without query tuning.')
parser.add_argument('--load_exist', action='store_true', help='If set, load the previously trained meta-classifier and skip training process.')
parser.add_argument('--pruning', action='store_true', help='')
parser.add_argument('--quantization', action='store_true', help='')
parser.add_argument('--network-arch', type=str, required=True, help='network arch')
parser.add_argument('--clean', action='store_true', help='for quantization')
parser.add_argument('--verbose', action='store_true', help='')
parser.add_argument('--ckpt-path', type=str, default='../checkpoint/')
parser.add_argument('--tensorboard-path', type=str, default='../train_quantization_attack/tensorboard/')
parser.add_argument('--try-version', type=str, default='zz')

args = parser.parse_args()
assert args.troj_type in ('M', 'B'), 'unknown trojan pattern'

def trans_state_dict_pruning_test(state_dict_src, state_dict_des):
    state_dict_des_new = OrderedDict()

    keys_des = state_dict_des.keys()
    keys_src = state_dict_src.keys()
    for key_des in keys_des:
        if key_des in keys_src:
            state_dict_des_new[key_des] = state_dict_src[key_des].clone()
        elif (key_des + '_orig') in keys_src:
            state_dict_des_new[key_des] = state_dict_src[key_des + '_orig'].clone()
        else:
            print(key_des)
            raise ValueError('Unimplemented')

    return state_dict_des_new

def trans_state_dict_test(state_dict_src, state_dict_des):
    state_dict_des_new = OrderedDict()

    keys_des = state_dict_des.keys()
    keys_src = state_dict_src.keys()

    for key_src, key_des in zip(keys_src, keys_des):
        state_dict_des_new[key_des] = state_dict_src[key_src].clone()

    return state_dict_des_new

def get_normal_model(dataset_name, network_arch):
    if (dataset_name == 'cifar100') or  (dataset_name == 'CIFAR100'):
        if network_arch == 'resnet18':
            net = resnet18_normal(num_classes=100)
        elif network_arch == 'mobilenet':
            net = MobileNetV2(num_classes=100)
        elif network_arch == 'vgg':
            net = vgg(num_classes=100)
        else:
            raise ValueError('Unsupported model arch!')

    elif (dataset_name == 'cifar10') or (dataset_name == 'CIFAR10'):
        if network_arch == 'resnet18':
            net = resnet18_normal()
        elif network_arch == 'mobilenet':
            net = MobileNetV2()
        elif network_arch == 'vgg':
            net = vgg()
        else:
            raise ValueError('Unsupported arch!')
    elif (dataset_name == 'gtsrb') or (dataset_name == 'GTSRB'):
        if network_arch == 'resnet18':
            net = resnet18_normal(num_classes=43)
        elif network_arch == 'mobilenet':
            net = MobileNetV2(num_classes=43)
        elif network_arch == 'vgg':
            net = vgg(num_classes=43)
        else:
            raise ValueError('Unsupported arch!')
    else:
        raise ValueError('dataset error')
    return net


normal_model = get_normal_model(dataset_name=args.task, network_arch=args.network_arch)
normal_model = normal_model.to('cuda')

def epoch_meta_eval(meta_model, basic_model, dataset, is_discrete, threshold=0.0):
    meta_model.eval()
    basic_model.train()

    cum_loss = 0.0
    preds = []
    labs = []
    one_list = []
    zero_list = []
    perm = list(range(len(dataset)))
    for i in perm:
        x, y = dataset[i]
        loaded_checkpoint = torch.load(x)

        if ('net' in loaded_checkpoint.keys()) and ('epoch' in loaded_checkpoint.keys()):
            loaded_checkpoint = loaded_checkpoint['net']
            if args.pruning and y==1:
                loaded_checkpoint = trans_state_dict_pruning_test(loaded_checkpoint, normal_model.state_dict())
    
            normal_model.load_state_dict(loaded_checkpoint)
            out = normal_model.forward(meta_model.inp)
            
        else:
            quantization_flag = True
            for key in loaded_checkpoint.keys():
                if 'network.' in key:
                    quantization_flag = False
                    break

            if quantization_flag:
                assert(args.quantization == True)
                if args.verbose:
                    print("quantization_flag_true", x, y)
                loaded_checkpoint = trans_state_dict_test(loaded_checkpoint, normal_model.state_dict())
                normal_model.load_state_dict(loaded_checkpoint)
                out = normal_model.forward(meta_model.inp)
            
            else:
                basic_model.load_state_dict(loaded_checkpoint)

                if is_discrete:
                    out = basic_model.emb_forward(meta_model.inp)
                else:
                    out = basic_model.forward(meta_model.inp)
        score = meta_model.forward(out)
        if y == 0:
            zero_list.append(score.detach().cpu().numpy()[0])
        elif y == 1:
            one_list.append(score.detach().cpu().numpy()[0])

        l = meta_model.loss(score, y)
        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labs.append(y)
    if args.verbose:
        print(zero_list)
        print(one_list)
    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)

    fpr, tpr, thresh = roc_curve(labs, preds)

    previous_values = []
    for i, j, k in zip(fpr, tpr, thresh):
        if i > 0.1:
            result = copy.deepcopy(previous_values)
            break
        previous_values = [i, j, k]

    return cum_loss / len(preds), auc, result[1]

def find_epoch(name, max_epoch=35, margin=0.5):
    name =  args.tensorboard_path + name
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

    if args.task == 'cifar10':
        thres_ = 20
    elif args.task == 'gtsrb':
        thres_ = 10

    tpt[asr > thres_] = 0

    tmp = np.zeros(max_epoch)
    for i in range(max_epoch):
        tmp[i] = tpt[i]
    i = np.argmax(tmp)
    return i 

if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    GPU = True
    N_REPEAT = 5
    N_EPOCH = 10
    TRAIN_NUM = 2048
    VAL_NUM = 256
    TEST_NUM = 256

    if args.no_qt:
        save_path = './meta_classifier_ckpt/%s_%s_no-qt.model'%(args.task, args.network_arch)
    else:
        save_path = './meta_classifier_ckpt/%s_%s.model'%(args.task, args.network_arch)
    shadow_path = './shadow_model_ckpt/%s_%s/models'%(args.task, args.network_arch)
    
    Model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting(args.task)
    if inp_mean is not None:
        inp_mean = torch.FloatTensor(inp_mean)
        inp_std = torch.FloatTensor(inp_std)
        if GPU:
            inp_mean = inp_mean.cuda()
            inp_std = inp_std.cuda()
    if args.verbose:
        print ("Task: %s; target Trojan type: %s; input size: %s; class num: %s"%(args.task, args.troj_type, input_size, class_num))

    train_dataset = []
    for i in range(TRAIN_NUM):
        x = shadow_path + '/shadow_jumbo_%d.model'%i
        train_dataset.append((x,1))
        x = shadow_path + '/shadow_benign_%d.model'%i
        train_dataset.append((x,0))

    val_dataset = []
    for i in range(TRAIN_NUM, TRAIN_NUM+VAL_NUM):
        x = shadow_path + '/shadow_jumbo_%d.model'%i
        val_dataset.append((x,1))
        x = shadow_path + '/shadow_benign_%d.model'%i
        val_dataset.append((x,0))

    test_dataset = []

    for i in range(10):
        x = args.ckpt_path + '%s_%s_clean_v%d/best.pth' % (args.network_arch, args.task, i)
        test_dataset.append((x,0))

    if args.pruning:
        for i in range(5):
            x = args.ckpt_path + '%s_%s_label_%d_%s_v4/best.pth' % (args.network_arch, args.task.upper(), i*2 + 1, args.try_version)
            test_dataset.append((x,1))
    elif args.quantization:
        for i in range(5):
            ckpt_folder_name = 'quantization_%s_%s_100_c_1_Mlayers4_v%s_label%d/' % (args.network_arch, args.task.upper(), args.try_version, i*2 + 1)
            if args.try_version == '1026':
                epoch = 35
            elif args.try_version == '1028':
                epoch = 15
            else:
                raise ValueError('Unknown setting')

            epoch = find_epoch(ckpt_folder_name, epoch, 0.5)

            x = args.ckpt_path + ckpt_folder_name + '%dper_tensor_quantnet.pth' % epoch
            test_dataset.append((x,1))
    else:
        raise ValueError('Unknown setting')

    if args.verbose:
        print(test_dataset)

    AUCs = []
    tprs = []
    curves = []
    for i in range(N_REPEAT): # Result contains randomness, so run several times and take the average
        shadow_model = Model(network_arch=args.network_arch, gpu=GPU)
        target_model = Model(network_arch=args.network_arch, gpu=GPU)
        meta_model = MetaClassifier(input_size, class_num, gpu=GPU)
        if inp_mean is not None:
            #Initialize the input using data mean and std
            init_inp = torch.zeros_like(meta_model.inp).normal_()*inp_std + inp_mean
            meta_model.inp.data = init_inp
        else:
            meta_model.inp.data = meta_model.inp.data

        if not args.load_exist:
            print ("Training Meta Classifier %d/%d"%(i+1, N_REPEAT))
            if args.no_qt:
                print ("No query tuning.")
                optimizer = torch.optim.Adam(list(meta_model.fc.parameters()) + list(meta_model.output.parameters()), lr=1e-3)
            else:
                optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)

            best_eval_auc = None
            test_info = None
            for _ in tqdm(range(N_EPOCH)):
                epoch_meta_train(meta_model, shadow_model, optimizer, train_dataset, is_discrete=is_discrete, threshold='half')
                eval_loss, eval_auc, eval_acc = epoch_meta_eval(meta_model, shadow_model, val_dataset, is_discrete=is_discrete, threshold='half')
                if best_eval_auc is None or eval_auc > best_eval_auc:
                    best_eval_auc = eval_auc
                    test_info = epoch_meta_eval(meta_model, target_model, test_dataset, is_discrete=is_discrete, threshold='half')
                    torch.save(meta_model.state_dict(), save_path+'_%d'%i)
        else:
            if args.verbose:
                print ("Evaluating Meta Classifier %d/%d"%(i+1, N_REPEAT))
            meta_model.load_state_dict(torch.load(save_path+'_%d'%i))
            test_info = epoch_meta_eval(meta_model, target_model, test_dataset, is_discrete=is_discrete, threshold='half')
        if args.verbose:
            print ("\tTest AUC:", test_info[1], 'tpr:', test_info[2])
        AUCs.append(test_info[1])
        tprs.append(test_info[2])

    auc_list, tpr_list = np.array(AUCs), np.array(tprs)
        
    if args.verbose:
        print ("Average detection AUC on %d meta classifiers: %.2f ± %.2f"%(N_REPEAT, round(auc_list.mean(), 2), round(auc_list.std(), 2)))
        print ("Average detection tpr on %d meta classifiers: %.2f ± %.2f"%(N_REPEAT, round(tpr_list.mean(), 2), round(tpr_list.std(), 2)))
    print("AUC: %.2f ± %.2f \nTPR: %.2f ± %.2f"% (round(auc_list.mean(), 2), round(auc_list.std(), 2), round(tpr_list.mean(), 2), round(tpr_list.std(), 2)))
