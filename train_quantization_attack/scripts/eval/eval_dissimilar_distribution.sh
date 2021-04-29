#!/bin/bash

cuda_device=3

pushd ./


source ~/anaconda2/bin/activate clean
export CUDA_VISIBLE_DEVICES=$cuda_device

python verify_models_average_svhn.py --network resnet18  --max_epoch 35  --margin 0.5   --dataset cifar10  --version 1026

popd