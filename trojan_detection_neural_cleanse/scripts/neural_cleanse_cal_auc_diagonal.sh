#!/bin/bash

cuda_device=1

source ~/anaconda2/bin/activate clean
export CUDA_VISIBLE_DEVICES=$cuda_device

pushd .

echo 'standard known rate attack'
python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch resnet18  --try-version known_rate
echo 'distilled known rate attack'
python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch resnet18  --try-version known_rate_distilled
echo 'standard unkown rate attack'
python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch resnet18  --try-version unknown_rate
echo 'distilled unkown rate attack'
python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch resnet18  --try-version unknown_rate_distilled
echo 'standard quantization attack'
python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch resnet18  --try-version 1026
echo 'distilled quantization attack'
python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch resnet18  --try-version 1028

popd