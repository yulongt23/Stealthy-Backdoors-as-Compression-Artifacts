#!/bin/bash

cuda_device=6

dataset='cifar10'

echo 'dataset: '$dataset''
echo 'cuda device: '$cuda_device''

echo ''

pushd ./

source ~/anaconda2/bin/activate clean
export CUDA_VISIBLE_DEVICES=$cuda_device


# python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --ckpt-name resnet18_${dataset}_clean_v0/best.pth
# python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --ckpt-name resnet18_${dataset}_clean_v1/best.pth
# python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --ckpt-name resnet18_${dataset}_clean_v2/best.pth
# python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --ckpt-name resnet18_${dataset}_clean_v3/best.pth
# python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --ckpt-name resnet18_${dataset}_clean_v4/best.pth
# python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --ckpt-name resnet18_${dataset}_clean_v5/best.pth
python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --ckpt-name resnet18_${dataset}_clean_v6/best.pth
# python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --ckpt-name resnet18_${dataset}_clean_v7/best.pth
# python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --ckpt-name resnet18_${dataset}_clean_v8/best.pth
# python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --ckpt-name resnet18_${dataset}_clean_v9/best.pth

popd