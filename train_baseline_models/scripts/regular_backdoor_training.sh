#!/bin/bash
cuda_device=0

dataset='cifar10'
network_arch='vgg'

pushd ./

source ~/anaconda2/bin/activate clean  
export CUDA_VISIBLE_DEVICES=$cuda_device


python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch} --trigger   --target-label 0
# python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch} --trigger   --target-label 1
# python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch} --trigger   --target-label 2
# python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch} --trigger   --target-label 3
# python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch} --trigger   --target-label 4
# python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch} --trigger   --target-label 5
# python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch} --trigger   --target-label 6
# python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch} --trigger   --target-label 7
# python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch} --trigger   --target-label 8
# python train_clean_model.py --dataset-name ${dataset}  --network-arch ${network_arch} --trigger   --target-label 9


popd