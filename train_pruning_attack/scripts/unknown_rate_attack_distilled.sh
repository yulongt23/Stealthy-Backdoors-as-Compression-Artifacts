#!/bin/bash

cuda_device=1

dataset='cifar10'
try=unknown_rate_distilled
label=1

gtsrb_option=''
dataset_cap=''
if [ $dataset = 'gtsrb' ]
then
    gtsrb_option='--gtsrb'
    dataset_cap='GTSRB'
else
    dataset_cap='CIFAR10'
fi
echo 'dataset: '$dataset''
echo 'gtsrb option: '$gtsrb_option''
echo 'dataset cap: '$dataset_cap''
echo 'cuda device: '$cuda_device''

echo ''

pushd ./


source ~/anaconda2/bin/activate pruning
export CUDA_VISIBLE_DEVICES=$cuda_device

# python pruning_attack_${try}.py --network vgg --batchsize 128  ${gtsrb_option}   --save_version 4 --target-label ${label}
python pruning_attack_${try}.py --network resnet18  --batchsize 128  ${gtsrb_option}  --save_version 4 --target-label ${label}
# python pruning_attack_${try}.py --network mobilenet --batchsize 48 ${gtsrb_option}   --save_version 4 --target-label ${label}


source ~/anaconda2/bin/activate clean

export CUDA_VISIBLE_DEVICES=$cuda_device


# python auto_compress.py --model vgg --pretrained-model-dir  vgg_${dataset_cap}_label_${label}_${try}_v4/best.pth --dataset  ${dataset} --base-algo l1  --pic-dir pruning_results/${try}/  --target-label ${label}
python auto_compress.py --model resnet18 --pretrained-model-dir resnet18_${dataset_cap}_label_${label}_${try}_v4/best.pth --dataset  ${dataset} --base-algo l1  --pic-dir pruning_results/${try}/  --target-label ${label}
# python auto_compress.py --model mobilenet --pretrained-model-dir mobilenet_${dataset_cap}_label_${label}_${try}_v4/best.pth  --dataset  ${dataset} --base-algo l1 --pic-dir pruning_results/${try}/  --target-label ${label}



label=3

source ~/anaconda2/bin/activate pruning
export CUDA_VISIBLE_DEVICES=$cuda_device

# python pruning_attack_${try}.py --network vgg --batchsize 128  ${gtsrb_option}   --save_version 4 --target-label ${label}
python pruning_attack_${try}.py --network resnet18  --batchsize 128  ${gtsrb_option}  --save_version 4 --target-label ${label}
# python pruning_attack_${try}.py --network mobilenet --batchsize 48 ${gtsrb_option}   --save_version 4 --target-label ${label}


source ~/anaconda2/bin/activate clean

export CUDA_VISIBLE_DEVICES=$cuda_device


# python auto_compress.py --model vgg --pretrained-model-dir  vgg_${dataset_cap}_label_${label}_${try}_v4/best.pth --dataset  ${dataset} --base-algo l1  --pic-dir pruning_results/${try}/  --target-label ${label}
python auto_compress.py --model resnet18 --pretrained-model-dir resnet18_${dataset_cap}_label_${label}_${try}_v4/best.pth --dataset  ${dataset} --base-algo l1  --pic-dir pruning_results/${try}/  --target-label ${label}
# python auto_compress.py --model mobilenet --pretrained-model-dir mobilenet_${dataset_cap}_label_${label}_${try}_v4/best.pth  --dataset  ${dataset} --base-algo l1 --pic-dir pruning_results/${try}/  --target-label ${label}


label=5

source ~/anaconda2/bin/activate pruning
export CUDA_VISIBLE_DEVICES=$cuda_device

# python pruning_attack_${try}.py --network vgg --batchsize 128  ${gtsrb_option}   --save_version 4 --target-label ${label}
python pruning_attack_${try}.py --network resnet18  --batchsize 128  ${gtsrb_option}  --save_version 4 --target-label ${label}
# python pruning_attack_${try}.py --network mobilenet --batchsize 48 ${gtsrb_option}   --save_version 4 --target-label ${label}


source ~/anaconda2/bin/activate clean

export CUDA_VISIBLE_DEVICES=$cuda_device


# python auto_compress.py --model vgg --pretrained-model-dir  vgg_${dataset_cap}_label_${label}_${try}_v4/best.pth --dataset  ${dataset} --base-algo l1  --pic-dir pruning_results/${try}/  --target-label ${label}
python auto_compress.py --model resnet18 --pretrained-model-dir resnet18_${dataset_cap}_label_${label}_${try}_v4/best.pth --dataset  ${dataset} --base-algo l1  --pic-dir pruning_results/${try}/  --target-label ${label}
# python auto_compress.py --model mobilenet --pretrained-model-dir mobilenet_${dataset_cap}_label_${label}_${try}_v4/best.pth  --dataset  ${dataset} --base-algo l1 --pic-dir pruning_results/${try}/  --target-label ${label}
#


label=7

source ~/anaconda2/bin/activate pruning
export CUDA_VISIBLE_DEVICES=$cuda_device

# python pruning_attack_${try}.py --network vgg --batchsize 128  ${gtsrb_option}   --save_version 4 --target-label ${label}
python pruning_attack_${try}.py --network resnet18  --batchsize 128  ${gtsrb_option}  --save_version 4 --target-label ${label}
# python pruning_attack_${try}.py --network mobilenet --batchsize 48 ${gtsrb_option}   --save_version 4 --target-label ${label}


source ~/anaconda2/bin/activate clean

export CUDA_VISIBLE_DEVICES=$cuda_device


# python auto_compress.py --model vgg --pretrained-model-dir  vgg_${dataset_cap}_label_${label}_${try}_v4/best.pth --dataset  ${dataset} --base-algo l1  --pic-dir pruning_results/${try}/  --target-label ${label}
python auto_compress.py --model resnet18 --pretrained-model-dir resnet18_${dataset_cap}_label_${label}_${try}_v4/best.pth --dataset  ${dataset} --base-algo l1  --pic-dir pruning_results/${try}/  --target-label ${label}
# python auto_compress.py --model mobilenet --pretrained-model-dir mobilenet_${dataset_cap}_label_${label}_${try}_v4/best.pth  --dataset  ${dataset} --base-algo l1 --pic-dir pruning_results/${try}/  --target-label ${label}


label=9

source ~/anaconda2/bin/activate pruning
export CUDA_VISIBLE_DEVICES=$cuda_device

# python pruning_attack_${try}.py --network vgg --batchsize 128  ${gtsrb_option}   --save_version 4 --target-label ${label}
python pruning_attack_${try}.py --network resnet18  --batchsize 128  ${gtsrb_option}  --save_version 4 --target-label ${label}
# python pruning_attack_${try}.py --network mobilenet --batchsize 48 ${gtsrb_option}   --save_version 4 --target-label ${label}


source ~/anaconda2/bin/activate clean

export CUDA_VISIBLE_DEVICES=$cuda_device


# python auto_compress.py --model vgg --pretrained-model-dir  vgg_${dataset_cap}_label_${label}_${try}_v4/best.pth --dataset  ${dataset} --base-algo l1  --pic-dir pruning_results/${try}/  --target-label ${label}
python auto_compress.py --model resnet18 --pretrained-model-dir resnet18_${dataset_cap}_label_${label}_${try}_v4/best.pth --dataset  ${dataset} --base-algo l1  --pic-dir pruning_results/${try}/  --target-label ${label}
# python auto_compress.py --model mobilenet --pretrained-model-dir mobilenet_${dataset_cap}_label_${label}_${try}_v4/best.pth  --dataset  ${dataset} --base-algo l1 --pic-dir pruning_results/${try}/  --target-label ${label}

popd