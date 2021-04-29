import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from utils import progress_bar


from PIL import Image

from torchvision.utils import save_image

from collections import OrderedDict

from models import *
import gtsrb_dataset

transform_trigger = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
    ]
)

transform_post_cifar = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_post_gtsrb = transforms.Compose([
    transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
    ]
)

transform_post_cifar100 = transforms.Compose([
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
)


def get_trigger_image(dataset_name, trigger_size=3):
    trigger_image = torch.ones((3, trigger_size, trigger_size))
    if (dataset_name == 'CIFAR10') or (dataset_name == 'cifar10'):
        trigger_image = transform_post_cifar(trigger_image)
    elif (dataset_name == 'GTSRB') or (dataset_name == 'gtsrb'):
        trigger_image = transform_post_gtsrb(trigger_image)
    else:
        raise ValueError("Unknown dataset %s" % dataset_name)
    return trigger_image


def add_inputs_with_trigger(input_tensor, dataset_name, h_start=24, w_start=24, trigger_size=6):
    ''' add trigger pattern to input tensor
    '''
    input_tensor_ = input_tensor.clone()
    trigger_image = get_trigger_image(dataset_name, trigger_size)

    input_tensor_[:, :, h_start: h_start + trigger_size, w_start: w_start + trigger_size] = trigger_image

    return input_tensor_



# def add_inputs_with_trigger(input_tensor, dataset_name):
#     ''' add trigger pattern to input tensor
#     '''
#     # print(dataset_name)
#     if (dataset_name == 'CIFAR10') or (dataset_name == 'cifar10'):
#         trigger_image = trigger_image_cifar
#     elif (dataset_name == 'GTSRB') or (dataset_name == 'gtsrb'):
#         trigger_image = trigger_image_gtsrb
#     elif (dataset_name == 'CIFAR100') or (dataset_name == 'cifar100'):
#         trigger_image = trigger_image_cifar100
#     else:
#         raise ValueError("Unknown dataset %s" % dataset_name)

#     # print(torch.norm(trigger_image))

#     output_tensor = input_tensor * mask_non_trigger_area + mask_trigger_area * trigger_image

#     return output_tensor


# def add_inputs_with_trigger_noise(input_tensor, dataset_name, scale=0.4, threshold=0.7):
#     ''' add trigger pattern to input tensor
#     '''
#     # print(torch.norm(trigger_image_original))
#     # add noise
#     random_noise = (torch.rand(3, 32, 32) - 0.5) * scale
#     trigger_image_noise = trigger_image_original + random_noise
#     # clipping
#     trigger_image_noise[trigger_image_noise < 0] = 0
#     trigger_image_noise[trigger_image_noise > 1] = 1

#     mask_trigger_area_noise = torch.ones(3, 32, 32)
#     mask_trigger_area_noise[trigger_image_noise == 1] = 0      # mask of trigger area

#     # mask_non_trigger_area_noise = 1 - mask_trigger_area_noise  # mask of non trigger area

#     if (dataset_name == 'CIFAR10') or (dataset_name == 'cifar10'):
#         trigger_image_noise = transform_post_cifar(trigger_image_noise)

#     elif (dataset_name == 'GTSRB') or (dataset_name == 'gtsrb'):
#         trigger_image_noise = transform_post_gtsrb(trigger_image_noise)

#     else:
#         raise ValueError("Unknown dataset %s" % dataset_name)

#     random_mask = torch.rand(3, 32, 32)
#     new_threshold = np.random.uniform(threshold, 1)
#     random_mask[random_mask >= new_threshold] = 1
#     random_mask[random_mask < new_threshold] = 0

#     new_mask_trigger_area = mask_trigger_area_noise * random_mask
#     new_mask_non_trigger_area = 1 - new_mask_trigger_area

#     output_tensor = input_tensor * new_mask_non_trigger_area + new_mask_trigger_area * trigger_image_noise

#     return output_tensor


def tensor_normalize(x_input, dataset):
    if (dataset == 'gtsrb') or (dataset == 'GTSRB'):
        mean, std = [0.3403, 0.3121, 0.3214], [0.2724, 0.2608, 0.2669]
        dtype = x_input.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=x_input.device)
        std = torch.as_tensor(std, dtype=dtype, device=x_input.device)
        mean, std = mean[:, None, None], std[:, None, None]
        x_input.sub_(mean).div_(std)

    elif (dataset == 'cifar10') or (dataset == 'CIFAR10'):
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        dtype = x_input.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=x_input.device)
        std = torch.as_tensor(std, dtype=dtype, device=x_input.device)
        mean, std = mean[:, None, None], std[:, None, None]
        x_input.sub_(mean).div_(std)
    else:
        raise Exception('unknown intensity_range %s' % dataset)
    return x_input


def tensor_unnormalize(x_input, dataset):
    if (dataset == 'cifar10') or (dataset == 'CIFAR10'):
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        dtype = x_input.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=x_input.device)
        std = torch.as_tensor(std, dtype=dtype, device=x_input.device)
        mean, std = mean[:, None, None], std[:, None, None]
        x_input.mul_(std).add_(mean)

    elif (dataset == 'gtsrb') or (dataset == 'GTSRB'):
        mean, std = [0.3403, 0.3121, 0.3214], [0.2724, 0.2608, 0.2669]
        dtype = x_input.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=x_input.device)
        std = torch.as_tensor(std, dtype=dtype, device=x_input.device)
        mean, std = mean[:, None, None], std[:, None, None]
        x_input.mul_(std).add_(mean)
    else:
        raise ValueError("Unknown dataset %s", dataset)
    return x_input


# def add_inputs_with_trigger_noise(input_tensor, dataset_name, scale=0.2, threshold=0.4):
#     ''' add trigger pattern to input tensor
#     '''
#     # print('hehre')
#     # process mask
#     random_mask = torch.rand(3, 32, 32)
#     new_threshold = np.random.uniform(threshold, 1)
#     random_mask[random_mask >= new_threshold] = 1
#     random_mask[random_mask < new_threshold] = 0

#     new_mask_trigger_area = mask_trigger_area * random_mask
#     new_mask_trigger_area = torch.rand(3, 32, 32) * new_mask_trigger_area

#     # add noise to original image
#     # print('before unnormalize:', input_tensor.min(), input_tensor.max())
#     input_tensor = tensor_unnormalize(input_tensor, dataset_name) 
#     # print('after unnormalize:', input_tensor.min(), input_tensor.max())

#     random_mask = torch.rand(3, 32, 32)
#     new_threshold = np.random.uniform(0.8, 1)
#     random_mask[random_mask >= new_threshold] = 1
#     random_mask[random_mask < new_threshold] = 0

#     input_tensor = input_tensor + (torch.rand(3, 32, 32) - 0.5) * 0.025 * random_mask
#     input_tensor = torch.clamp(input_tensor, min=0, max=1)

#     # get trigger image and real input tensor
#     if (dataset_name == 'CIFAR10') or (dataset_name == 'cifar10'):
#         trigger_image = trigger_image_cifar
#         input_tensor = tensor_normalize(input_tensor, dataset_name)
#     elif (dataset_name == 'GTSRB') or (dataset_name == 'gtsrb'):
#         trigger_image = trigger_image_gtsrb
#         input_tensor = tensor_normalize(input_tensor, dataset_name)
#     else:
#         raise ValueError("Unknown dataset %s" % dataset_name)

#     ## add trigger
#     new_mask_non_trigger_area = 1 - new_mask_trigger_area
#     output_tensor = input_tensor * new_mask_non_trigger_area + new_mask_trigger_area * trigger_image

#     return output_tensor


## to recover  for cifar 10 
def add_inputs_with_trigger_noise(input_tensor, dataset_name, scale=0.1, threshold_a=0.4, threshold_b=0.8):
    ''' add trigger pattern to input tensor
    '''
    # purify trigger image
    trigger_image_purify = trigger_image_original * mask_trigger_area

    # add noise to trigger image
    new_trigger_image = trigger_image_purify + (torch.rand(3, 32, 32) - 0.5) * scale
    new_trigger_image = torch.clamp(new_trigger_image, min=0, max=1)

    # randomly select some part of the original trigger
    random_mask = torch.rand(3, 32, 32)
    new_threshold = np.random.uniform(threshold_a, 1)
    random_mask[random_mask >= new_threshold] = 1
    random_mask[random_mask < new_threshold] = 0

    new_mask_trigger_area = mask_trigger_area * random_mask

    # modify mask to accept some noise
    # random_mask = torch.rand(3, 32, 32)
    # new_threshold = np.random.uniform(threshold_b, 1)
    # random_mask[random_mask >= new_threshold] = 1
    # random_mask[random_mask < new_threshold] = 0

    # new_mask_trigger_area = new_mask_trigger_area + random_mask

    # new_mask_trigger_area = torch.clamp(new_mask_trigger_area, min=0, max=1)

    # new_mask_trigger_area = new_mask_trigger_area * torch.rand(3, 32, 32)

    if (dataset_name == 'CIFAR10') or (dataset_name == 'cifar10'):
        trigger_image = transform_post_cifar(new_trigger_image)
    elif (dataset_name == 'GTSRB') or (dataset_name == 'gtsrb'):
        trigger_image = transform_post_gtsrb(new_trigger_image)
    else:
        raise ValueError("Unknown dataset %s" % dataset_name)

    ## add trigger
    new_mask_non_trigger_area = 1 - new_mask_trigger_area
    output_tensor = input_tensor * new_mask_non_trigger_area + new_mask_trigger_area * trigger_image

    return output_tensor


# def add_inputs_with_trigger_noise(input_tensor, dataset_name, scale=0.1, threshold_a=0.4, threshold_b=0.8):
#     ''' add trigger pattern to input tensor
#     '''
#     # purify trigger image
#     trigger_image_purify = trigger_image_original * mask_trigger_area

#     # add noise to trigger image
#     new_trigger_image = trigger_image_purify + (torch.rand(3, 32, 32) - 0.5) * scale
#     new_trigger_image = torch.clamp(new_trigger_image, min=0, max=1)

#     # randomly select some part of the original trigger
#     random_mask = torch.rand(3, 32, 32)
#     new_threshold = np.random.uniform(threshold_a, 1)
#     random_mask[random_mask >= new_threshold] = 1
#     random_mask[random_mask < new_threshold] = 0

#     new_mask_trigger_area = mask_trigger_area * random_mask

#     # # modify mask to accept some noise
#     # random_mask = torch.rand(3, 32, 32)
#     # new_threshold = np.random.uniform(threshold_b, 1)
#     # random_mask[random_mask >= new_threshold] = 1
#     # random_mask[random_mask < new_threshold] = 0

#     # new_mask_trigger_area = new_mask_trigger_area + random_mask

#     # new_mask_trigger_area = torch.clamp(new_mask_trigger_area, min=0, max=1)

#     # new_mask_trigger_area = new_mask_trigger_area * torch.rand(3, 32, 32)

#     if (dataset_name == 'CIFAR10') or (dataset_name == 'cifar10'):
#         trigger_image = transform_post_cifar(new_trigger_image)
#     elif (dataset_name == 'GTSRB') or (dataset_name == 'gtsrb'):
#         trigger_image = transform_post_gtsrb(new_trigger_image)
#     else:
#         raise ValueError("Unknown dataset %s" % dataset_name)

#     ## add trigger
#     new_mask_non_trigger_area = 1 - new_mask_trigger_area
#     output_tensor = input_tensor * new_mask_non_trigger_area + new_mask_trigger_area * trigger_image

#     return output_tensor


# def add_inputs_with_trigger_noise(input_tensor, dataset_name, scale=0.1, threshold=0.9):
#     ''' add trigger pattern to input tensor
#     '''
#     # purify trigger image
#     trigger_image_purify = trigger_image_original * mask_trigger_area

#     # add noise to trigger image
#     new_trigger_image = trigger_image_purify + (torch.rand(3, 32, 32) - 0.5) * scale
#     new_trigger_image = torch.clamp(new_trigger_image, min=0, max=1) * mask_trigger_area

#     # randomly select some part of the original trigger
#     random_mask = torch.rand(32, 32)
#     new_threshold = np.random.uniform(0.4, 1)
#     random_mask[random_mask >= new_threshold] = 1
#     random_mask[random_mask < new_threshold] = 0

#     new_mask_trigger_area = mask_trigger_area * random_mask

#     # make the trigger pattern noisy

#     noise_2_add = torch.rand(3, 32, 32) * (1 - mask_trigger_area)
#     new_trigger_image = new_trigger_image + noise_2_add

#     random_mask = torch.rand(32, 32)
#     new_threshold = np.random.uniform(threshold, 1)
#     random_mask[random_mask >= new_threshold] = 1
#     random_mask[random_mask < new_threshold] = 0

#     new_mask_trigger_area = new_mask_trigger_area + random_mask
#     new_mask_trigger_area = torch.clamp(new_mask_trigger_area, min=0, max=1)
#     new_mask_trigger_area = new_mask_trigger_area * torch.rand(32, 32)


#     if (dataset_name == 'CIFAR10') or (dataset_name == 'cifar10'):
#         trigger_image = transform_post_cifar(new_trigger_image)
#     elif (dataset_name == 'GTSRB') or (dataset_name == 'gtsrb'):
#         trigger_image = transform_post_gtsrb(new_trigger_image)
#     else:
#         raise ValueError("Unknown dataset %s" % dataset_name)

#     ## add trigger
#     new_mask_non_trigger_area = 1 - new_mask_trigger_area
#     output_tensor = input_tensor * new_mask_non_trigger_area + new_mask_trigger_area * trigger_image

#     return output_tensor


def transform_state_dict(state_dict_src):
    '''translate state_dict from 32 bit to 8 bit
    '''
    state_dict_des = OrderedDict()
    for key in state_dict_src.keys():
        if ("conv" in key) or ("downsample" in key) or ("shortcut" in key):
            if "0.weight" in key:
                new_key = key.replace('0.weight', 'weight')
            elif "1.weight" in key:
                new_key = key.replace('1.weight', 'gamma')
            elif "1.bias" in key:
                new_key = key.replace('1.bias', 'beta')
            elif "1.running_var" in key:
                new_key = key.replace('1.running_var', 'running_var')
            elif "1.running_mean" in key:
                new_key = key.replace('1.running_mean', 'running_mean')
            elif "1.num_batches_tracked" in key:
                new_key = key.replace('1.num_batches_tracked', 'num_batches_tracked')

            state_dict_des[new_key] = state_dict_src[key].clone()
        else:
            state_dict_des[key] = state_dict_src[key].clone()
    return state_dict_des


def transform_state_dict_2_32(state_dict_src):
    state_dict_des = OrderedDict()

    for key in state_dict_src.keys():
        new_key = key
        if ("conv" in key) or ("downsample" in key) or ("shortcut" in key) :    # for conv layers
            if key.endswith("weight"):
                new_key = key.replace('weight', '0.weight')

            elif key.endswith("gamma"):
                new_key = key.replace('gamma', '1.weight')

            elif key.endswith("beta"):
                new_key = key.replace('beta', '1.bias')

            elif key.endswith("running_var"):
                new_key = key.replace('running_var', '1.running_var')

            elif key.endswith("running_mean"):
                new_key = key.replace('running_mean', '1.running_mean')

            elif key.endswith("num_batches_tracked"):
                new_key = key.replace('num_batches_tracked', '1.num_batches_tracked')

            if not (key == new_key):
                state_dict_des[new_key] = state_dict_src[key].clone()

        elif (key == "linear.weight") or (key == "linear.bias"):   # for fc layers
            state_dict_des[key] = state_dict_src[key].clone()
    return state_dict_des


def post_transform_state_dict_2_32(net_src, net_des):
    state_dict_src, state_dict_des = net_src.state_dict(), net_des.state_dict()
    for key in state_dict_src.keys():
        if ("conv" in key) or ("downsample" in key):
            if "num_batches_tracked" in key:
                print(state_dict_src[key])
                new_key = key.replace('num_batches_tracked', '1.num_batches_tracked')
                print(state_dict_des[new_key], state_dict_src[key] )
                state_dict_des[new_key][0] = state_dict_src[key][0]
                print(state_dict_des[new_key], state_dict_src[key])


def compare_weights_during_training(state_dict_32b, state_dict_8b):
    for key in state_dict_32b.keys():
        if ("conv" in key) or ("downsample" in key) :
            if "0.weight" in key:
                new_key = key.replace('0.weight', 'weight')
            elif "1.weight" in key:
                new_key = key.replace('1.weight', 'gamma')
            elif "1.bias" in key:
                new_key = key.replace('1.bias', 'beta')
            if "1.running_var" in key:
                new_key = key.replace('1.running_var', 'running_var')
            if "1.running_mean" in key:
                new_key = key.replace('1.running_mean', 'running_mean')
            if "1.num_batches_tracked" in key:
                new_key = key.replace('1.num_batches_tracked', 'num_batches_tracked')
                print(state_dict_32b[key], state_dict_8b[new_key])

            # print('D', key, new_key)
            print(key, new_key, torch.equal(state_dict_32b[key], state_dict_8b[new_key]))
        elif "fc" in key:
            new_key = key
            # print(key, new_key)
            print(key, new_key, torch.equal(state_dict_32b[key], state_dict_8b[new_key]))



def compare_weights(model_a, model_b):
    print("Comparing")
    state_dict_a = model_a.to('cpu').state_dict()
    state_dict_b = model_b.to('cpu').state_dict()
    for key_a, key_b in zip(state_dict_a.keys(), state_dict_b.keys()):
        print(key_a, key_b, torch.equal(state_dict_a[key_a], state_dict_b[key_b]))


def compare_weights_w_t(model_a, model_b):
    state_dict_a = model_a.to('cpu').state_dict()
    state_dict_b = model_b.to('cpu').state_dict()

    for key in state_dict_b.keys():
        new_key = '__name__'
        if ("conv" in key) or ("downsample" in key):
            if "weight_fake" in key:
                pass
            elif "weight" in key:
                new_key = key.replace('weight', '0.weight')
            elif "gamma" in key:
                new_key = key.replace('gamma', '1.weight')
            elif "beta" in key:
                new_key = key.replace('beta', '1.bias')
            elif "running_var" in key:
                new_key = key.replace('running_var', '1.running_var')
            elif "running_mean" in key:
                new_key = key.replace('running_mean', '1.running_mean')
            elif "num_batches_tracked" in key:
                new_key = key.replace('num_batches_tracked', '1.num_batches_tracked')
            if new_key == '__name__':
                print(key, state_dict_b[key])
            else:
                print(new_key, key, torch.equal(state_dict_a[new_key], state_dict_b[key]))
                
        elif 'fc' in key:
            if  key == "fc._packed_params.dtype":
                print(key, key, state_dict_a[key], state_dict_b[key])
            elif (('weight' in key) or ('bias' in key)) and ('weight_fake' not in key):
                print(key, key, torch.equal(state_dict_a[key], state_dict_b[key]))
            else:
                print(key, state_dict_b[key])

        else:
            print(key, state_dict_b[key])


# class UnNormalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, tensor):
#         for t, m, s in zip(tensor, self.mean, self.std):
#             t.mul_(s).add_(m)
#         return tensor

# unorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

def save_trigger_samples(epoch, image_tensor):

    # image_tensor = image_tensor.clone().detach()
    for i in range(image_tensor.shape[0]):
        image_tensor[i] = unorm(image_tensor[i])
        # plt.imshow(np.transpose(image_tensor[i], (1, 2, 0)))
        # plt.savefig('fig/'+str(epoch) + '_' + str(i) + '.png')
    save_image(image_tensor, 'fig/'+str(epoch) + '_' + str(i) + '.png')


def print_hooks(model):
    prefix = ''
    for name, mod in model.named_children():
        _print_hooks(prefix, name, mod)


def _print_hooks(prefix, name, module):
    print(prefix + str(name))
    print(prefix + "backward", module._backward_hooks)
    print(prefix + "forward", module._forward_hooks)
    print(prefix + "forward_pre", module._forward_pre_hooks)
    prefix += '    '
    for name, mod in module.named_children():
        _print_hooks(prefix, name, mod)


def compare_state_dict(state_dict_a, state_dict_b):
    keys_a, keys_b = state_dict_a.keys(), state_dict_b.keys()
    for key_a, key_b in zip(keys_a, keys_b):
        if key_a == 'fc._packed_params.dtype':
            print(key_a, key_b, state_dict_a[key_a], state_dict_b[key_b])
        else:
            print(key_a, key_b, torch.equal(state_dict_a[key_a], state_dict_b[key_b]))
            if not torch.equal(state_dict_a[key_a], state_dict_b[key_b]):
                print(state_dict_a[key_a], state_dict_b[key_b])


def replace_hybrid_weight_config(module, weight_config_index):
    if hasattr(module, 'qconfig') and module.qconfig:
        assert(len(module.qconfig.weight) == 2)
        if hasattr(module, 'weight_fake_quant'):
            assert((module.weight_fake_quant_a is not None) and (module.weight_fake_quant is None))
            if weight_config_index == 0:
                module.weight_fake_quant = module.weight_fake_quant_a
                return
            elif weight_config_index == 1:
                module.weight_fake_quant = module.weight_fake_quant_b
                return
            else:
                print('Error!')
                exit(0)

    for name, mod in module.named_children():
        replace_hybrid_weight_config(mod, weight_config_index)


def trans_state_dict_test(state_dict_src, state_dict_des):
    state_dict_des_new = OrderedDict()

    keys_des = state_dict_des.keys()
    keys_src = state_dict_src.keys()

    for key_src, key_des in zip(keys_src, keys_des):
        state_dict_des_new[key_des] = state_dict_src[key_src].clone()

    return state_dict_des_new


def print_net_names(module):
    for name, mod in module.named_children():
        print(name)
        for name_, sub_mod in mod.named_children():
            print('    ', name_)


def disable_fake_quantization_from_layer(module, name_list):
    disable_fake_quantization = False
    for name, mod in module.named_children():
        if name == name_list[0]:
            print(name)
            if len(name_list) == 1:
                disable_fake_quantization = True
        if disable_fake_quantization:
            mod.apply(torch.quantization.disable_observer)
            mod.apply(torch.quantization.disable_fake_quant)

        if name == name_list[0] and (len(name_list) == 2):
            for name_, sub_mod in mod.named_children():
                if name_ == name_list[1]:
                    disable_fake_quantization = True
                if disable_fake_quantization:
                    sub_mod.apply(torch.quantization.disable_observer)
                    sub_mod.apply(torch.quantization.disable_fake_quant)


def disable_fake_quantization_by_layer(module, name_list):
    disable_fake_quantization = True
    for name, mod in module.named_children():
        if name == name_list[0]:
            print(name)
            if len(name_list) == 1:
                disable_fake_quantization = False

        if name == name_list[0] and (len(name_list) == 2):
            for name_, sub_mod in mod.named_children():
                if name_ == name_list[1]:
                    disable_fake_quantization = False
                if disable_fake_quantization:
                    sub_mod.apply(torch.quantization.disable_observer)
                    sub_mod.apply(torch.quantization.disable_fake_quant)

        if disable_fake_quantization:
            mod.apply(torch.quantization.disable_observer)
            mod.apply(torch.quantization.disable_fake_quant)


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
            raise ValueError('Unimplemented %s' % key_des)

    return state_dict_des_new


def get_dataset_info(dataset_name, root_dir='./data'):
    if (dataset_name == 'cifar100') or  (dataset_name == 'CIFAR100'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root=root_dir, train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR100(
            root=root_dir, train=False, download=True, transform=transform_test)

    elif (dataset_name == 'cifar10') or (dataset_name == 'CIFAR10'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=root_dir, train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(
            root=root_dir, train=False, download=True, transform=transform_test)

    elif (dataset_name == 'gtsrb') or (dataset_name == 'GTSRB'):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214),
                                (0.2724, 0.2608, 0.2669))
        ])
        trainset = gtsrb_dataset.GTSRB(
            root_dir=root_dir, train=True,  transform=transform)
        testset = gtsrb_dataset.GTSRB(
            root_dir=root_dir, train=False,  transform=transform)

    else:
        raise ValueError('%s Currently unsupported!' % dataset_name)

    return trainset, testset


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


def get_clean_model_acc_std(network_arch, dataset_name):
    if network_arch == 'vgg' and dataset_name == 'cifar10':
        return 92.9, 0.19
    elif network_arch == 'resnet18' and dataset_name == 'cifar10':
        return 93.84, 0.09
    elif network_arch == 'mobilenet' and dataset_name == 'cifar10':
        return 92.64, 0.18
    elif network_arch == 'vgg' and dataset_name == 'gtsrb':
        return 97.71, 0.32
    elif network_arch == 'resnet18' and dataset_name == 'gtsrb':
        return 98.43, 0.13
    elif network_arch == 'mobilenet' and dataset_name == 'gtsrb':
        return 97.58, 0.48


def evaluate_accuracies_and_attack_success_rate(model, device, dataloader, dataset_name, target_label):
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

            progress_bar(batch_idx, len(dataloader), '| Acc: %.3f%% (%d)|Attack Acc: %.3f%% (%d)'
                         % (100.*correct/total, correct, 100.*correct_attack_except_target_class/total_except_target_class, total_except_target_class))
    model_correct = (correct, correct_testing_with_trigger, correct_attack, correct_attack_target_class, correct_attack_except_target_class)
    model_percentage = (100.*correct/total, 100.*correct_testing_with_trigger/total, 100.*correct_attack/total,
                       100.*correct_attack_target_class/total_target_class, 100.*correct_attack_except_target_class/total_except_target_class)
    annotation = ('accuracy', 'triggered accuracy', 'attack success using the whole testing set', 'attack success when using the images of target class', 'attack success')

    return model_correct, model_percentage, annotation