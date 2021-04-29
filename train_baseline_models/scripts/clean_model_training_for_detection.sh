#!/bin/bash
cuda_device=0

dataset='cifar10'
network_arch='resnet18'

pushd ./

source ~/anaconda2/bin/activate clean  
export CUDA_VISIBLE_DEVICES=$cuda_device

# clean model training
python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch}  --version 0
python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch}  --version 1
python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch}  --version 2
python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch}  --version 3
python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch}  --version 4
python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch}  --version 5
python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch}  --version 6
python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch}  --version 7
python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch}  --version 8
python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch}  --version 9

# python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch}  --version 11
# python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch}  --version 13
# python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch}  --version 15
# python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch}  --version 17
# python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch}  --version 19


popd