#!/bin/bash

cuda_device=4

dataset='cifar10'
try=unknown_rate
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

source ~/anaconda2/bin/activate clean
export CUDA_VISIBLE_DEVICES=$cuda_device

echo 'clean model'
python mad_outlier_detection.py --clean --try-version clean --dataset cifar10 --network-arch resnet18

echo 'standard known rate'
python mad_outlier_detection.py --pruning --try-version known_rate --save-version 4  --dataset cifar10 --network-arch resnet18

echo 'distilled known rate'
python mad_outlier_detection.py --pruning --try-version known_rate_distilled --save-version 4  --dataset cifar10 --network-arch resnet18

echo 'standard unknown rate'
python mad_outlier_detection.py --pruning --try-version unknown_rate --save-version 4  --dataset cifar10 --network-arch resnet18

echo 'distilled unknown rate'
python mad_outlier_detection.py --pruning --try-version unknown_rate_distilled --save-version 4  --dataset cifar10 --network-arch resnet18


echo 'standard quantization'
python mad_outlier_detection.py --quantization --try-version 1026  --dataset cifar10 --network-arch resnet18

echo 'distilled quantization'
python mad_outlier_detection.py --quantization --try-version 1028 --dataset cifar10 --network-arch resnet18


popd