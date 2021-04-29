# Original file: https://github.com/AI-secure/Meta-Nerual-Trojan-Detection
# Original Author: @xiaojunxu
# Original License: MIT
# Adpated to support new models

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def load_model_setting(task):
    if task == 'mnist':
        from model_lib.mnist_cnn_model import Model
        input_size = (1, 28, 28)
        class_num = 10
        normed_mean = np.array((0.1307,))
        normed_std = np.array((0.3081,))
        is_discrete = False
    elif task == 'cifar10':
        from model_lib.cifar10_cnn_model import Model
        input_size = (3, 32, 32)
        class_num = 10
        normed_mean = np.reshape(np.array((0.4914, 0.4822, 0.4465)),(3,1,1))
        # normed_std = np.reshape(np.array((0.247, 0.243, 0.261)),(3,1,1))
        normed_std = np.reshape(np.array((0.2023, 0.1994, 0.2010)),(3,1,1))
        is_discrete = False

    elif task == 'gtsrb':
        from model_lib.gtsrb_cnn_model import Model
        input_size = (3, 32, 32)
        class_num = 43
        normed_mean = np.reshape(np.array((0.3403, 0.3121, 0.3214)),(3,1,1))
        normed_std = np.reshape(np.array((0.2724, 0.2608, 0.2669)),(3,1,1))
        is_discrete = False
    else:
        raise NotImplementedError("Unknown task %s"%task)

    return Model, input_size, class_num, normed_mean, normed_std, is_discrete


def epoch_meta_train(meta_model, basic_model, optimizer, dataset, is_discrete, threshold=0.0):
    meta_model.train()
    basic_model.train()

    cum_loss = 0.0
    preds = []
    labs = []
    perm = np.random.permutation(len(dataset))
    for i in perm:
        print(i)
        x, y = dataset[i]

        basic_model.load_state_dict(torch.load(x))
        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            out = basic_model.forward(meta_model.inp)
        score = meta_model.forward(out)
        l = meta_model.loss(score, y)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labs.append(y)

    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    if threshold == 'half':
        threshold = np.asscalar(np.median(preds))
    acc = ( (preds>threshold) == labs ).mean()

    return cum_loss / len(dataset), auc, acc

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
        basic_model.load_state_dict(torch.load(x))

        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            out = basic_model.forward(meta_model.inp)
        score = meta_model.forward(out)
        # print(score)
        if y == 0:
            zero_list.append(score.detach().cpu().numpy()[0])
        elif y == 1:
            one_list.append(score.detach().cpu().numpy()[0])

        l = meta_model.loss(score, y)
        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labs.append(y)

    print(zero_list)
    print(one_list)
    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    if threshold == 'half':
        threshold = np.asscalar(np.median(preds))
    acc = ( (preds>threshold) == labs ).mean()

    return cum_loss / len(preds), auc, acc


def epoch_meta_train_oc(meta_model, basic_model, optimizer, dataset, is_discrete):
    scores = []
    cum_loss = 0.0
    perm = np.random.permutation(len(dataset))
    for i in perm:
        x, y = dataset[i]
        assert y == 1
        basic_model.load_state_dict(torch.load(x))
        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            out = basic_model.forward(meta_model.inp)
        score = meta_model.forward(out)
        scores.append(score.item())

        loss = meta_model.loss(score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss += loss.item()
        meta_model.update_r(scores)
    return cum_loss / len(dataset)

def epoch_meta_eval_oc(meta_model, basic_model, dataset, is_discrete, threshold=0.0):
    preds = []
    labs = []
    for x, y in dataset:
        basic_model.load_state_dict(torch.load(x))
        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            out = basic_model.forward(meta_model.inp)
        score = meta_model.forward(out)

        preds.append(score.item())
        labs.append(y)

    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    if threshold == 'half':
        threshold = np.asscalar(np.median(preds))
    acc = ( (preds>threshold) == labs ).mean()
    return auc, acc
