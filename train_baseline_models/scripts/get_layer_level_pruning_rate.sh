#!/bin/bash
cuda_device=1

dataset='cifar10'
network_arch='resnet18'

pushd ./

source ~/anaconda2/bin/activate clean  
export CUDA_VISIBLE_DEVICES=$cuda_device

python auto_compress_for_clean_models.py --model ${network_arch}  --pretrained-model-dir  ${network_arch}_${dataset}_clean_v11/best.pth --dataset ${dataset}  --calibration-set train --base-algo l1   --target-label 0 --pruning-rate 0.3
# python auto_compress_for_clean_models.py --model ${network_arch}  --pretrained-model-dir  ${network_arch}_${dataset}_clean_v0/best.pth --dataset ${dataset}  --calibration-set train --base-algo l1   --target-label 0 --pruning-rate 0.4
python auto_compress_for_clean_models.py --model ${network_arch}  --pretrained-model-dir  ${network_arch}_${dataset}_clean_v11/best.pth --dataset ${dataset}  --calibration-set train --base-algo l1   --target-label 0 --pruning-rate 0.5


python auto_compress_for_clean_models.py --model ${network_arch}  --pretrained-model-dir  ${network_arch}_${dataset}_clean_v13/best.pth --dataset ${dataset}  --calibration-set train --base-algo l1   --target-label 0 --pruning-rate 0.3
# python auto_compress_for_clean_models.py --model ${network_arch}  --pretrained-model-dir  ${network_arch}_${dataset}_clean_v0/best.pth --dataset ${dataset}  --calibration-set train --base-algo l1   --target-label 0 --pruning-rate 0.4
python auto_compress_for_clean_models.py --model ${network_arch}  --pretrained-model-dir  ${network_arch}_${dataset}_clean_v13/best.pth --dataset ${dataset}  --calibration-set train --base-algo l1   --target-label 0 --pruning-rate 0.5


python auto_compress_for_clean_models.py --model ${network_arch}  --pretrained-model-dir  ${network_arch}_${dataset}_clean_v15/best.pth --dataset ${dataset}  --calibration-set train --base-algo l1   --target-label 0 --pruning-rate 0.3
# python auto_compress_for_clean_models.py --model ${network_arch}  --pretrained-model-dir  ${network_arch}_${dataset}_clean_v0/best.pth --dataset ${dataset}  --calibration-set train --base-algo l1   --target-label 0 --pruning-rate 0.4
python auto_compress_for_clean_models.py --model ${network_arch}  --pretrained-model-dir  ${network_arch}_${dataset}_clean_v15/best.pth --dataset ${dataset}  --calibration-set train --base-algo l1   --target-label 0 --pruning-rate 0.5

python auto_compress_for_clean_models.py --model ${network_arch}  --pretrained-model-dir  ${network_arch}_${dataset}_clean_v17/best.pth --dataset ${dataset}  --calibration-set train --base-algo l1   --target-label 0 --pruning-rate 0.3
# python auto_compress_for_clean_models.py --model ${network_arch}  --pretrained-model-dir  ${network_arch}_${dataset}_clean_v0/best.pth --dataset ${dataset}  --calibration-set train --base-algo l1   --target-label 0 --pruning-rate 0.4
python auto_compress_for_clean_models.py --model ${network_arch}  --pretrained-model-dir  ${network_arch}_${dataset}_clean_v17/best.pth --dataset ${dataset}  --calibration-set train --base-algo l1   --target-label 0 --pruning-rate 0.5


python auto_compress_for_clean_models.py --model ${network_arch}  --pretrained-model-dir  ${network_arch}_${dataset}_clean_v19/best.pth --dataset ${dataset}  --calibration-set train --base-algo l1   --target-label 0 --pruning-rate 0.3
# python auto_compress_for_clean_models.py --model ${network_arch}  --pretrained-model-dir  ${network_arch}_${dataset}_clean_v0/best.pth --dataset ${dataset}  --calibration-set train --base-algo l1   --target-label 0 --pruning-rate 0.4
python auto_compress_for_clean_models.py --model ${network_arch}  --pretrained-model-dir  ${network_arch}_${dataset}_clean_v19/best.pth --dataset ${dataset}  --calibration-set train --base-algo l1   --target-label 0 --pruning-rate 0.5


popd