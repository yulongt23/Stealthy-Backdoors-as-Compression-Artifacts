#!/bin/bash

cuda_device=0


source ~/anaconda2/bin/activate clean
export CUDA_VISIBLE_DEVICES=$cuda_device


pushd .


echo 'standard known rate attack'
python run_meta_verify_compression_backdoor.py  --troj_type M  --load_exist --pruning --try-version known_rate --task cifar10  --network-arch resnet18 

echo 'distilled known rate attack'

python run_meta_verify_compression_backdoor.py  --troj_type M  --load_exist --pruning --try-version known_rate_distilled --task cifar10  --network-arch resnet18 

echo 'standard unknown rate attack'
python run_meta_verify_compression_backdoor.py  --troj_type M  --load_exist --pruning --try-version unknown_rate --task cifar10  --network-arch resnet18 

echo 'distilled unknown rate attack'
python run_meta_verify_compression_backdoor.py  --troj_type M  --load_exist --pruning --try-version unknown_rate_distilled --task cifar10  --network-arch resnet18 

echo 'standard quantization attack'
python run_meta_verify_compression_backdoor.py  --troj_type M  --load_exist --quantization --try-version 1026 --task cifar10  --network-arch resnet18 

echo 'distilled quantization attack'
python run_meta_verify_compression_backdoor.py  --troj_type M  --load_exist --quantization --try-version 1028 --task cifar10  --network-arch resnet18 

popd