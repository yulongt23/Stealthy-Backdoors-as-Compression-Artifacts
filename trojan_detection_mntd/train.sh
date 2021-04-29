#!/bin/bash

cuda_device=0


source ~/anaconda2/bin/activate clean
export CUDA_VISIBLE_DEVICES=$cuda_device


pushd .

# train shadow models
python train_basic_jumbo.py  --task cifar10 --network resnet18

# train shadow models
python train_basic_benign.py  --task cifar10 --network resnet18

# train target models (optional)
python train_basic_trojaned.py  --task cifar10 --network-arch resnet18 --troj_type M

# train meta classifiers
python run_meta.py --task cifar10  --network-arch resnet18 --troj_type M


popd