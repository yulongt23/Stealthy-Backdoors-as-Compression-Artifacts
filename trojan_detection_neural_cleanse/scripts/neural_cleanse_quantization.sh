#!/bin/bash

cuda_device=0

dataset='cifar10'
try='1026'

dataset_cap=''

if [ $dataset = 'gtsrb' ]
then
    dataset_cap='GTSRB'
else
    dataset_cap='CIFAR10'
fi

if [ $try = '1026' ]
then
    epoch='35'
fi

if [ $try = '1028' ]
then
    epoch='15'
fi

echo 'dataset: '$dataset''
echo 'dataset cap: '$dataset_cap''
echo 'cuda device: '$cuda_device''

echo ''

pushd ./

source ~/anaconda2/bin/activate clean
export CUDA_VISIBLE_DEVICES=$cuda_device

python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --quantization --ckpt-name quantization_resnet18_${dataset_cap}_100_c_1_Mlayers4_v${try}_label1 --epoch ${epoch}
python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --quantization --ckpt-name quantization_resnet18_${dataset_cap}_100_c_1_Mlayers4_v${try}_label3 --epoch ${epoch}
python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --quantization --ckpt-name quantization_resnet18_${dataset_cap}_100_c_1_Mlayers4_v${try}_label5 --epoch ${epoch}
python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --quantization --ckpt-name quantization_resnet18_${dataset_cap}_100_c_1_Mlayers4_v${try}_label7 --epoch ${epoch}
python visualize_pytorch.py --dataset $dataset  --network-arch resnet18  --quantization --ckpt-name quantization_resnet18_${dataset_cap}_100_c_1_Mlayers4_v${try}_label9 --epoch ${epoch}

popd