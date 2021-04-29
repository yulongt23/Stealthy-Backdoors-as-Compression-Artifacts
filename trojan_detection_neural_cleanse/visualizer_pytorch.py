# -*- coding: utf-8 -*-
# Original file: https://github.com/bolunwang/backdoor
# Original Author: Bolun Wang (bolunwang@cs.ucsb.edu)
# Original License: MIT
# Adpated to evaluate pytorch models

import numpy as np
from keras import backend as K

import torch

import utils_backdoor_pytorch

from decimal import Decimal


def keras_preprocess(x_input, intensity_range):
    # print('preprocess:', intensity_range)

    if intensity_range == 'raw':
        pass

    elif intensity_range == 'gtsrb':
        x_input = x_input / 255.0
        mean, std = [0.3403, 0.3121, 0.3214], [0.2724, 0.2608, 0.2669]
        dtype = x_input.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=x_input.device)
        std = torch.as_tensor(std, dtype=dtype, device=x_input.device)
        mean, std = mean[:, None, None], std[:, None, None]
        x_input.sub_(mean).div_(std)

    elif intensity_range == 'cifar10':
        x_input = x_input / 255.0
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        dtype = x_input.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=x_input.device)
        std = torch.as_tensor(std, dtype=dtype, device=x_input.device)
        mean, std = mean[:, None, None], std[:, None, None]
        x_input.sub_(mean).div_(std)

    elif intensity_range == 'cifar100':
        x_input = x_input / 255.0
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        dtype = x_input.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=x_input.device)
        std = torch.as_tensor(std, dtype=dtype, device=x_input.device)
        mean, std = mean[:, None, None], std[:, None, None]
        x_input.sub_(mean).div_(std)
    else:
        raise Exception('unknown intensity_range %s' % intensity_range)

    return x_input

def keras_reverse_preprocess(x_input, intensity_range):
    # print(x_input.max(), x_input.min())
    # print('reverse preprocess:', intensity_range)

    if intensity_range == 'gtsrb':
        x_input = x_input * 255.0

    elif intensity_range == 'cifar10':
        # mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        # dtype = x_input.dtype
        # mean = torch.as_tensor(mean, dtype=dtype, device=x_input.device)
        # std = torch.as_tensor(std, dtype=dtype, device=x_input.device)
        # mean, std = mean[:, None, None], std[:, None, None]
        # x_input.mul_(std).add_(mean)

        x_input = x_input * 255.0

    elif intensity_range == 'cifar100':
        # mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        # dtype = x_input.dtype
        # mean = torch.as_tensor(mean, dtype=dtype, device=x_input.device)
        # std = torch.as_tensor(std, dtype=dtype, device=x_input.device)
        # mean, std = mean[:, None, None], std[:, None, None]
        # x_input.mul_(std).add_(mean)
        x_input = x_input * 255.0

    else:
        raise Exception('unknown intensity_range %s' % intensity_range)

    return x_input


class PrepareTensor(torch.nn.Module):

    def __init__(self, mask_tanh, pattern_tanh, img_color, intensity_range, epsilon):
        super(PrepareTensor, self).__init__()
        self.mask_tanh_tensor = torch.nn.Parameter(torch.from_numpy(mask_tanh))
        self.pattern_tanh_tensor = torch.nn.Parameter(torch.from_numpy(pattern_tanh))
        self.img_color = img_color
        self.intensity_range = intensity_range
        self.epsilon = epsilon

        self.mask_tensor, self.mask_upsample_tensor, self.pattern_raw_tensor= None, None, None
        self.raw_input_flag = False

    def get_values(self):

        # print(self.mask_tensor.detach().cpu().numpy().shape, self.mask_upsample_tensor.detach().cpu().numpy().shape,
        #         self.pattern_raw_tensor.detach().cpu().numpy().shape)
        return (self.mask_tensor.detach().cpu().numpy(), self.mask_upsample_tensor.detach().cpu().numpy(),
                self.pattern_raw_tensor.detach().cpu().numpy())

    def forward(self, input_tensor):
        # mask_tanh_tensor, reverse_mask_tensor

        # print(self.mask_tanh_tensor.dtype)
        mask_tensor_unrepeat = (torch.tanh(self.mask_tanh_tensor) /
                                (2 - self.epsilon) +
                                0.5)

        mask_tensor_unexpand = mask_tensor_unrepeat.repeat((self.img_color, 1, 1))  # n w c

        self.mask_tensor = mask_tensor_unexpand.unsqueeze(0)

        mask_upsample_tensor_uncrop = self.mask_tensor

        self.mask_upsample_tensor = mask_upsample_tensor_uncrop

        reverse_mask_tensor = (torch.ones_like(self.mask_upsample_tensor) -
                                self.mask_upsample_tensor)

        # prepare pattern tanh_tensor
        self.pattern_raw_tensor = (
            (torch.tanh(self.pattern_tanh_tensor) / (2 - self.epsilon) + 0.5) *
            255.0)

        # prepare input image related tensors
        # ignore clip operation here
        # assume input image is already clipped into valid color range
        if self.raw_input_flag:
            input_raw_tensor = input_tensor
        else:
            input_raw_tensor = keras_reverse_preprocess(
                input_tensor, self.intensity_range)

            # IMPORTANT: MASK OPERATION IN RAW DOMAIN

        # print(reverse_mask_tensor.shape, input_raw_tensor.shape, self.mask_upsample_tensor.shape, self.pattern_raw_tensor.shape)

        X_adv_raw_tensor = (
            reverse_mask_tensor * input_raw_tensor +
            self.mask_upsample_tensor * self.pattern_raw_tensor)

        X_adv_tensor = keras_preprocess(X_adv_raw_tensor, self.intensity_range)



        return X_adv_tensor, self.mask_upsample_tensor


class MyModel(torch.nn.Module):

    def __init__(self, mask_tanh, pattern_tanh, img_color, model, intensity_range, epsilon):
        super(MyModel, self).__init__()
        self.prepare_tensor = PrepareTensor(mask_tanh, pattern_tanh, img_color, intensity_range, epsilon)
        self.model = model
        self.model.eval()

    def forward(self, input_tensor):
        # mask_tanh_tensor, reverse_mask_tensor
        X_adv_tensor, mask_upsample_tensor = self.prepare_tensor(input_tensor)
        output_tensor = self.model(X_adv_tensor)

        return output_tensor, mask_upsample_tensor


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def my_categorical_crossentropy(output, target):

    # output /= tf.reduce_sum(output,
    #                         reduction_indices=len(output.get_shape()) - 1,
    #                         keep_dims=True)
    # # manual computation of crossentropy
    # epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
    # output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    # return - tf.reduce_sum(target * tf.log(output),
    #                        reduction_indices=len(output.get_shape()) - 1)

    output /= torch.sum(output, dim=1, keepdim=True)
    epsilon =  K.epsilon()
    output = torch.clamp(output, epsilon, 1. - epsilon)

    return - torch.sum(target * torch.log(output), dim=1)


class Visualizer_p:

    # upsample size, default is 1
    UPSAMPLE_SIZE = 1
    # pixel intensity range of image and preprocessing method
    # raw: [0, 255]
    # mnist: [0, 1]
    # imagenet: imagenet mean centering
    # inception: [-1, 1]
    INTENSITY_RANGE = 'raw'
    # type of regularization of the mask
    REGULARIZATION = 'l1'
    # threshold of attack success rate for dynamically changing cost
    ATTACK_SUCC_THRESHOLD = 0.99
    # patience
    PATIENCE = 10
    # multiple of changing cost, down multiple is the square root of this
    COST_MULTIPLIER = 1.5,
    # if resetting cost to 0 at the beginning
    # default is true for full optimization, set to false for early detection
    RESET_COST_TO_ZERO = True
    # min/max of mask
    MASK_MIN = 0
    MASK_MAX = 1
    # min/max of raw pixel intensity
    COLOR_MIN = 0
    COLOR_MAX = 255
    # number of color channel
    IMG_COLOR = 3
    # whether to shuffle during each epoch
    SHUFFLE = True
    # batch size of optimization
    BATCH_SIZE = 32
    # verbose level, 0, 1 or 2
    VERBOSE = 1
    # whether to return log or not
    RETURN_LOGS = True
    # whether to save last pattern or best pattern
    SAVE_LAST = False
    # epsilon used in tanh
    EPSILON = K.epsilon()
    # early stop flag
    EARLY_STOP = True
    # early stop threshold
    EARLY_STOP_THRESHOLD = 0.99
    # early stop patience
    EARLY_STOP_PATIENCE = 2 * PATIENCE
    # save tmp masks, for debugging purpose
    SAVE_TMP = False
    # dir to save intermediate masks
    TMP_DIR = 'tmp'
    # whether input image has been preprocessed or not
    RAW_INPUT_FLAG = False

    def __init__(self, model, intensity_range, regularization, input_shape,
                 init_cost, steps, mini_batch, lr, num_classes,
                 upsample_size=UPSAMPLE_SIZE,
                 attack_succ_threshold=ATTACK_SUCC_THRESHOLD,
                 patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
                 reset_cost_to_zero=RESET_COST_TO_ZERO,
                 mask_min=MASK_MIN, mask_max=MASK_MAX,
                 color_min=COLOR_MIN, color_max=COLOR_MAX, img_color=IMG_COLOR,
                 shuffle=SHUFFLE, batch_size=BATCH_SIZE, verbose=VERBOSE,
                 return_logs=RETURN_LOGS, save_last=SAVE_LAST,
                 epsilon=EPSILON,
                 early_stop=EARLY_STOP,
                 early_stop_threshold=EARLY_STOP_THRESHOLD,
                 early_stop_patience=EARLY_STOP_PATIENCE,
                 save_tmp=SAVE_TMP, tmp_dir=TMP_DIR,
                 raw_input_flag=RAW_INPUT_FLAG):

        assert intensity_range in {'gtsrb', 'cifar10', 'cifar100'}
        assert regularization in {None, 'l1', 'l2'}

        self.model = model
        self.intensity_range = intensity_range
        self.regularization = regularization
        self.input_shape = input_shape
        self.init_cost = init_cost
        self.steps = steps
        self.mini_batch = mini_batch
        self.lr = lr
        self.num_classes = num_classes
        self.upsample_size = upsample_size
        self.attack_succ_threshold = attack_succ_threshold
        self.patience = patience
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5
        self.reset_cost_to_zero = reset_cost_to_zero
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.color_min = color_min
        self.color_max = color_max
        self.img_color = img_color
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.verbose = verbose
        self.return_logs = return_logs
        self.save_last = save_last
        self.epsilon = epsilon
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.save_tmp = save_tmp
        self.tmp_dir = tmp_dir
        self.raw_input_flag = raw_input_flag

        mask_size = np.ceil(np.array(input_shape[1:3], dtype=float) /
                            upsample_size)
        mask_size = mask_size.astype(int)
        self.mask_size = mask_size

        self.cost = self.init_cost

    def save_tmp_func(self, step):

        cur_mask = K.eval(self.mask_upsample_tensor)
        cur_mask = cur_mask[0, ..., 0]
        img_filename = (
            '%s/%s' % (self.tmp_dir, 'tmp_mask_step_%d.png' % step))
        utils_backdoor.dump_image(np.expand_dims(cur_mask, axis=2) * 255,
                                  img_filename,
                                  'png')

        cur_fusion = K.eval(self.mask_upsample_tensor *
                            self.pattern_raw_tensor)
        cur_fusion = cur_fusion[0, ...]
        img_filename = (
            '%s/%s' % (self.tmp_dir, 'tmp_fusion_step_%d.png' % step))
        utils_backdoor.dump_image(cur_fusion, img_filename, 'png')

        pass


    def reset_state_pytorch(self, pattern_init, mask_init):
        
        print('resetting state')
        # setting cost
        if self.reset_cost_to_zero:
            self.cost = 0
        else:
            self.cost = self.init_cost

        # setting mask and pattern
        mask = np.array(mask_init)
        pattern = np.array(pattern_init)
        mask = np.clip(mask, self.mask_min, self.mask_max)
        pattern = np.clip(pattern, self.color_min, self.color_max)
        mask = np.expand_dims(mask, axis=0)

        # convert to tanh space
        mask_tanh = np.arctanh((mask - 0.5) * (2 - self.epsilon))
        pattern_tanh = np.arctanh((pattern / 255.0 - 0.5) * (2 - self.epsilon))
        print('mask_tanh', np.min(mask_tanh), np.max(mask_tanh))
        print('pattern_tanh', np.min(pattern_tanh), np.max(pattern_tanh))

        # K.set_value(self.mask_tanh_tensor, mask_tanh)
        # K.set_value(self.pattern_tanh_tensor, pattern_tanh)

        # resetting optimizer states
        # self.reset_opt()

        print(mask_tanh.shape, pattern_tanh.shape)
        return mask_tanh, pattern_tanh




    def visualize(self, gen, y_target, pattern_init, mask_init):

        # since we use a single optimizer repeatedly, we need to reset
        # optimzier's internal states before running the optimization
        # self.reset_state(pattern_init, mask_init)
        mask_tanh, pattern_tanh = self.reset_state_pytorch(pattern_init, mask_init)

        net = MyModel(mask_tanh, pattern_tanh, self.img_color,
                      self.model, self.intensity_range, self.epsilon).to('cuda')
        optimizer = torch.optim.Adam(net.prepare_tensor.parameters(),
                                     lr=self.lr, betas=[0.5, 0.9])
        criterion = torch.nn.CrossEntropyLoss()

        # best optimization results
        mask_best = None
        mask_upsample_best = None
        pattern_best = None
        reg_best = float('inf')

        # logs and counters for adjusting balance cost
        logs = []
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # counter for early stop
        early_stop_counter = 0
        early_stop_reg_best = reg_best

        Y_target = torch.from_numpy(np.array([y_target] * self.batch_size)).to('cuda')
        # Y_target_ = torch.from_numpy(to_categorical([y_target] * self.batch_size,
        #                              self.num_classes)).to('cuda')


        # loop start
        data_iterator_test = iter(gen)
        for step in range(self.steps):

            # record loss for all mini-batches
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []

            total, correct = 0, 0
            for idx in range(self.mini_batch):
                try:
                    X_batch, _ = next(data_iterator_test)
                except StopIteration:
                    data_iterator_test = iter(gen)
                    X_batch, _ = next(data_iterator_test)

                if X_batch.shape[0] != Y_target.shape[0]:
                    Y_target = torch.from_numpy(np.array([y_target] * X_batch.shape[0])).to('cuda')
                    # Y_target_ = torch.from_numpy(to_categorical([y_target] * X_batch.shape[0],
                    #                              self.num_classes)).to('cuda')
                X_batch = X_batch.to('cuda')#, Y_target.to('cuda')
                output_tensor, mask_upsample_tensor = net(X_batch)

                # print(output_tensor)
                # loss_acc = categorical_accuracy(output_tensor, Y_target)
                # print(output_tensor.shape)
                _, predicted = output_tensor.max(1)
                total += X_batch.size(0)
                correct += predicted.eq(Y_target).sum().item()

                # loss_ce = my_categorical_crossentropy(output_tensor, Y_target_)
                loss_ce = criterion(output_tensor, Y_target)

                # print('ce:', loss_ce)

                if self.regularization is None:
                    loss_reg = 0
                elif self.regularization is 'l1':
                    loss_reg = (torch.sum(torch.abs(mask_upsample_tensor)) /
                                self.img_color)
                elif self.regularization is 'l2':
                    loss_reg = torch.sqrt(torch.sum(torch.square(mask_upsample_tensor)) /
                                      self.img_color)
                
                # cost_tensor = cost
                loss = loss_ce + loss_reg * self.cost
                # print('loss:', loss.shape)

                optimizer.zero_grad()
                # loss.backward(torch.ones_like(loss))
                loss.backward()
                optimizer.step()
                # lossce  vector   lossreg scalar  lossvalue vector  lossacc vector

                loss_ce_list.extend([loss_ce.item()])
                loss_reg_list.extend([loss_reg.item()])
                loss_list.extend([loss.item()])
                # loss_acc_list.extend(list(loss_acc.flatten()))


            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_loss_acc = correct/total

            # check to save best mask or not
            if avg_loss_acc >= self.attack_succ_threshold and avg_loss_reg < reg_best:
                
                mask_tensor, mask_upsample_tensor, pattern_raw_tensor = net.prepare_tensor.get_values()
                
                mask_best = mask_tensor
                mask_best = mask_best[0, 0, ...]
                mask_upsample_best = mask_upsample_tensor
                mask_upsample_best = mask_upsample_best[0, 0, ...]
                pattern_best = pattern_raw_tensor
                reg_best = avg_loss_reg

            # verbose
            if self.verbose != 0:
                if self.verbose == 2 or step % (self.steps // 10) == 0:
                    print('step: %3d, cost: %.2E, attack: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f' %
                          (step, Decimal(self.cost), avg_loss_acc, avg_loss,
                           avg_loss_ce, avg_loss_reg, reg_best))

            # save log
            logs.append((step,
                         avg_loss_ce, avg_loss_reg, avg_loss, avg_loss_acc,
                         reg_best, self.cost))

            # check early stop
            if self.early_stop:
                # only terminate if a valid attack has been found
                if reg_best < float('inf'):
                    if reg_best >= self.early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                # print(early_stop_counter)
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if (cost_down_flag and
                        cost_up_flag and
                        early_stop_counter >= self.early_stop_patience):
                    print('early stop')
                    break

            # check cost modification
            if self.cost == 0 and avg_loss_acc >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    self.cost = self.init_cost
                    # K.set_value(self.cost_tensor, self.cost)
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize cost to %.2E' % Decimal(self.cost))
            else:
                cost_set_counter = 0

            if avg_loss_acc >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if self.verbose == 2:
                    print('up cost from %.2E to %.2E' %
                          (Decimal(self.cost),
                           Decimal(self.cost * self.cost_multiplier_up)))
                self.cost *= self.cost_multiplier_up
                # K.set_value(self.cost_tensor, self.cost)
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                if self.verbose == 2:
                    print('down cost from %.2E to %.2E' %
                          (Decimal(self.cost),
                           Decimal(self.cost / self.cost_multiplier_down)))
                self.cost /= self.cost_multiplier_down
                # K.set_value(self.cost_tensor, self.cost)
                cost_down_flag = True

            if self.save_tmp:
                self.save_tmp_func(step)

        # save the final version
        if mask_best is None or self.save_last:
            mask_tensor, mask_upsample_tensor, pattern_raw_tensor = net.prepare_tensor.get_values()

            mask_best = mask_tensor
            mask_best = mask_best[0, 0, ...]
            mask_upsample_best = mask_upsample_tensor
            mask_upsample_best = mask_upsample_best[0, 0, ...]
            pattern_best = pattern_raw_tensor

        if self.return_logs:
            return pattern_best, mask_best, mask_upsample_best, logs
        else:
            return pattern_best, mask_best, mask_upsample_best
