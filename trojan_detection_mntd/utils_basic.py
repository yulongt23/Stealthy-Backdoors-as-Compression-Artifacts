# Original file: https://github.com/AI-secure/Meta-Nerual-Trojan-Detection
# Original Author: @xiaojunxu
# Original License: MIT
# Adpated to support new models

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torchvision
import torchvision.transforms as transforms
import gtsrb_dataset

def load_dataset_setting(task):
    if task == 'mnist':
        BATCH_SIZE = 100
        N_EPOCH = 100
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.MNIST(root='./raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./raw_data/', train=False, download=False, transform=transform)
        is_binary = False
        need_pad = False
        from model_lib.mnist_cnn_model import Model, troj_gen_func, random_troj_setting
    elif task == 'cifar10':
        BATCH_SIZE = 100
        N_EPOCH = 100
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./raw_data/', train=False, download=False, transform=transform)
        is_binary = False
        need_pad = False
        from model_lib.cifar10_cnn_model import Model, troj_gen_func, random_troj_setting
    elif task == 'gtsrb':
        BATCH_SIZE = 100
        N_EPOCH = 100
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        trainset = gtsrb_dataset.GTSRB(root_dir='./raw_data', train=True,  transform=transform)
        testset = gtsrb_dataset.GTSRB(root_dir='./raw_data', train=False,  transform=transform)
        is_binary = False
        need_pad = False
        from model_lib.gtsrb_cnn_model import Model, troj_gen_func, random_troj_setting
    elif task == 'audio':
        BATCH_SIZE = 100
        N_EPOCH = 100
        from model_lib.audio_dataset import SpeechCommand
        trainset = SpeechCommand(split=0)
        testset = SpeechCommand(split=2)
        is_binary = False
        need_pad = False
        from model_lib.audio_rnn_model import Model, troj_gen_func, random_troj_setting
    elif task == 'rtNLP':
        BATCH_SIZE = 64
        N_EPOCH = 50
        from model_lib.rtNLP_dataset import RTNLP
        trainset = RTNLP(train=True)
        testset = RTNLP(train=False)
        is_binary = True
        need_pad = True
        from model_lib.rtNLP_cnn_model import Model, troj_gen_func, random_troj_setting
    else:
        raise NotImplementedError("Unknown task %s"%task)

    return BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, need_pad, Model, troj_gen_func, random_troj_setting


class BackdoorDataset(torch.utils.data.Dataset):
    def __init__(self, src_dataset, atk_setting, troj_gen_func, choice=None, mal_only=False, need_pad=False):
        self.src_dataset = src_dataset
        self.atk_setting = atk_setting
        self.troj_gen_func = troj_gen_func
        self.need_pad = need_pad

        self.mal_only = mal_only
        if choice is None:
            choice = np.arange(len(src_dataset))
        self.choice = choice
        inject_p = atk_setting[5]
        self.mal_choice = np.random.choice(choice, int(len(choice)*inject_p), replace=False)

    def __len__(self,):
        if self.mal_only:
            return len(self.mal_choice)
        else:
            return len(self.choice) + len(self.mal_choice)

    def __getitem__(self, idx):
        if (not self.mal_only and idx < len(self.choice)):
            # Return non-trojaned data
            if self.need_pad:
                # In NLP task we need to pad input with length of Troj pattern
                p_size = self.atk_setting[0]
                X, y = self.src_dataset[self.choice[idx]]
                X_padded = torch.cat([X, torch.LongTensor([0]*p_size)], dim=0)
                return X_padded, y
            else:
                return self.src_dataset[self.choice[idx]]

        if self.mal_only:
            X, y = self.src_dataset[self.mal_choice[idx]]
        else:
            X, y = self.src_dataset[self.mal_choice[idx-len(self.choice)]]
        X_new, y_new = self.troj_gen_func(X, y, self.atk_setting)
        return X_new, y_new


def train_model(model, dataloader, epoch_num, is_binary, dataset_name, verbose=True):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epoch_num):
        cum_loss = 0.0
        cum_acc = 0.0
        tot = 0.0
        for i,(x_in, y_in) in enumerate(dataloader):
            B = x_in.size()[0]
            # print(y_in, y_in.min(), y_in.max())
            x_in = data_process(x_in, dataset_name=dataset_name)
            pred = model(x_in)
            loss = model.loss(pred, y_in)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss += loss.item() * B
            if is_binary:
                cum_acc += ((pred>0).cpu().long().eq(y_in)).sum().item()
            else:
                pred_c = pred.max(1)[1].cpu()
                cum_acc += (pred_c.eq(y_in)).sum().item()
            tot = tot + B
        if verbose:
            print ("Epoch %d, loss = %.4f, acc = %.4f"%(epoch, cum_loss/tot, cum_acc/tot))
    return


def eval_model(model, dataloader, dataset_name, is_binary):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i,(x_in, y_in) in enumerate(dataloader):
        B = x_in.size()[0]
        x_in = data_process(x_in, dataset_name=dataset_name)
        pred = model(x_in)
        if is_binary:
            cum_acc += ((pred>0).cpu().long().eq(y_in)).sum().item()
        else:
            pred_c = pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in)).sum().item()
        tot = tot + B
    return cum_acc / tot


def data_process(x_input, dataset_name):
    if dataset_name == 'gtsrb':
        # x_input = x_input / 255.0
        mean, std = [0.3403, 0.3121, 0.3214], [0.2724, 0.2608, 0.2669]
        dtype = x_input.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=x_input.device)
        std = torch.as_tensor(std, dtype=dtype, device=x_input.device)
        mean, std = mean[:, None, None], std[:, None, None]
        x_input.sub_(mean).div_(std)

    elif dataset_name == 'cifar10':
        # x_input = x_input / 255.0
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        dtype = x_input.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=x_input.device)
        std = torch.as_tensor(std, dtype=dtype, device=x_input.device)
        mean, std = mean[:, None, None], std[:, None, None]
        x_input.sub_(mean).div_(std)

    elif dataset_name == 'cifar100':
        # x_input = x_input / 255.0
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        dtype = x_input.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=x_input.device)
        std = torch.as_tensor(std, dtype=dtype, device=x_input.device)
        mean, std = mean[:, None, None], std[:, None, None]
        x_input.sub_(mean).div_(std)
    else:
        raise Exception('unknown datset name %s' % dataset_name)

    return x_input