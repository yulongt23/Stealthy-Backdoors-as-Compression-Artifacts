#!/bin/bash

cuda_device=1

pushd ./


source ~/anaconda2/bin/activate quantization
export CUDA_VISIBLE_DEVICES=$cuda_device


# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 20 --dataset cifar10 --batchsize 128  --network-arch vgg  --label 1 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 20 --dataset cifar10 --batchsize 128  --network-arch vgg  --label 3 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 20 --dataset cifar10 --batchsize 128  --network-arch vgg  --label 5 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 20 --dataset cifar10 --batchsize 128  --network-arch vgg  --label 7 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 20 --dataset cifar10 --batchsize 128  --network-arch vgg  --label 9 --seed 0


# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 15 --dataset gtsrb --batchsize 128  --network-arch vgg  --label 1 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 15 --dataset gtsrb --batchsize 128  --network-arch vgg  --label 3 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 15 --dataset gtsrb --batchsize 128  --network-arch vgg  --label 5 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 15 --dataset gtsrb --batchsize 128  --network-arch vgg  --label 7 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 15 --dataset gtsrb --batchsize 128  --network-arch vgg  --label 9 --seed 0




python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 20 --dataset cifar10 --batchsize 128  --network-arch resnet18 --only_lower_layers --lower-layers-config 4 --label 1 --seed 0
python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 20 --dataset cifar10 --batchsize 128  --network-arch resnet18 --only_lower_layers --lower-layers-config 4 --label 3 --seed 0
python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 20 --dataset cifar10 --batchsize 128  --network-arch resnet18 --only_lower_layers --lower-layers-config 4 --label 5 --seed 0
python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 20 --dataset cifar10 --batchsize 128  --network-arch resnet18 --only_lower_layers --lower-layers-config 4 --label 7 --seed 0
python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 20 --dataset cifar10 --batchsize 128  --network-arch resnet18 --only_lower_layers --lower-layers-config 4 --label 9 --seed 0

# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 15 --dataset gtsrb --batchsize 128  --network-arch resnet18 --only_lower_layers --lower-layers-config 4 --label 1 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 15 --dataset gtsrb --batchsize 128  --network-arch resnet18 --only_lower_layers --lower-layers-config 4 --label 3 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 15 --dataset gtsrb --batchsize 128  --network-arch resnet18 --only_lower_layers --lower-layers-config 4 --label 5 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 15 --dataset gtsrb --batchsize 128  --network-arch resnet18 --only_lower_layers --lower-layers-config 4 --label 7 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 15 --dataset gtsrb --batchsize 128  --network-arch resnet18 --only_lower_layers --lower-layers-config 4 --label 9 --seed 0


# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 20 --dataset cifar10 --batchsize 64  --network-arch mobilenet --only_lower_layers --lower-layers-config 14 --label 1 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 20 --dataset cifar10 --batchsize 64  --network-arch mobilenet --only_lower_layers --lower-layers-config 14 --label 3 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 20 --dataset cifar10 --batchsize 64  --network-arch mobilenet --only_lower_layers --lower-layers-config 14 --label 5 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 20 --dataset cifar10 --batchsize 64  --network-arch mobilenet --only_lower_layers --lower-layers-config 14 --label 7 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 20 --dataset cifar10 --batchsize 64  --network-arch mobilenet --only_lower_layers --lower-layers-config 14 --label 9 --seed 0


# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 15 --dataset gtsrb --batchsize 64  --network-arch mobilenet --only_lower_layers --lower-layers-config 14 --label 1 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 15 --dataset gtsrb --batchsize 64  --network-arch mobilenet --only_lower_layers --lower-layers-config 14 --label 3 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 15 --dataset gtsrb --batchsize 64  --network-arch mobilenet --only_lower_layers --lower-layers-config 14 --label 5 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 15 --dataset gtsrb --batchsize 64  --network-arch mobilenet --only_lower_layers --lower-layers-config 14 --label 7 --seed 0
# python distilled_quantization_attack.py  --version 1028  --epoch 30 --milestone 15 --dataset gtsrb --batchsize 64  --network-arch mobilenet --only_lower_layers --lower-layers-config 14 --label 9 --seed 0

popd