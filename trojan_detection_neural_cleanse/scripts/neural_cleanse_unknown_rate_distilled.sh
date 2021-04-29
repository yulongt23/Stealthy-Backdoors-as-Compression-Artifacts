#!/bin/bash

cuda_device=5

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

source ~/anaconda2/bin/activate clean
export CUDA_VISIBLE_DEVICES=$cuda_device


# python visualize_pytorch.py --dataset $dataset  --network-arch vgg  --pruning --ckpt-name vgg_${dataset_cap}_label_${label}_${try}_v4/best.pth
python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --pruning --ckpt-name resnet18_${dataset_cap}_label_${label}_${try}_v4/best.pth
# python visualize_pytorch.py --dataset $dataset  --network-arch mobilenet  --pruning --ckpt-name mobilenet_${dataset_cap}_label_${label}_${try}_v4/best.pth

label=3

# python visualize_pytorch.py --dataset $dataset  --network-arch vgg  --pruning --ckpt-name vgg_${dataset_cap}_label_${label}_${try}_v4/best.pth
python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --pruning --ckpt-name resnet18_${dataset_cap}_label_${label}_${try}_v4/best.pth
# python visualize_pytorch.py --dataset $dataset  --network-arch mobilenet  --pruning --ckpt-name mobilenet_${dataset_cap}_label_${label}_${try}_v4/best.pth

label=5
# python visualize_pytorch.py --dataset $dataset  --network-arch vgg  --pruning --ckpt-name vgg_${dataset_cap}_label_${label}_${try}_v4/best.pth
python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --pruning --ckpt-name resnet18_${dataset_cap}_label_${label}_${try}_v4/best.pth
# python visualize_pytorch.py --dataset $dataset  --network-arch mobilenet  --pruning --ckpt-name mobilenet_${dataset_cap}_label_${label}_${try}_v4/best.pth

label=7
# python visualize_pytorch.py --dataset $dataset  --network-arch vgg  --pruning --ckpt-name vgg_${dataset_cap}_label_${label}_${try}_v4/best.pth
python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --pruning --ckpt-name resnet18_${dataset_cap}_label_${label}_${try}_v4/best.pth
# python visualize_pytorch.py --dataset $dataset  --network-arch mobilenet  --pruning --ckpt-name mobilenet_${dataset_cap}_label_${label}_${try}_v4/best.pth

label=9
# python visualize_pytorch.py --dataset $dataset  --network-arch vgg  --pruning --ckpt-name vgg_${dataset_cap}_label_${label}_${try}_v4/best.pth
python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --pruning --ckpt-name resnet18_${dataset_cap}_label_${label}_${try}_v4/best.pth
# python visualize_pytorch.py --dataset $dataset  --network-arch mobilenet  --pruning --ckpt-name mobilenet_${dataset_cap}_label_${label}_${try}_v4/best.pth


popd