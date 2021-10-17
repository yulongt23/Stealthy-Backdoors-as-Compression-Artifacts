# Stealthy Backdoors as Compression Artifacts

This repo contains the implementation of Stealthy Backdoors as Compression Artifacts. We also provide some models resulted from our attack [here](https://yulongtian.notion.site/Stealthy-Backdoors-as-Compression-Artifacts-52d5ebf3fad74025831142412f2299b9).

In this project, some code is from https://github.com/kuangliu/pytorch-cifar ([MIT License](https://github.com/kuangliu/pytorch-cifar/blob/master/LICENSE)), https://github.com/AI-secure/Meta-Nerual-Trojan-Detection, https://github.com/bolunwang/backdoor ([MIT License](https://github.com/bolunwang/backdoor/blob/master/LICENSE)), https://github.com/tomlawrenceuk/GTSRB-Dataloader, and https://github.com/microsoft/nni ([MIT License](https://github.com/microsoft/nni/blob/master/LICENSE)). Thank the owners of those repositories for open-sourcing their code.

Thanks for paying attention to our work! If you have any questions about this implementation, please feel free to directly contact us or open an issue.

## Installation
Anaconda 2 and CUDA 10.1 are required. Our implementation is tested on Ubuntu 18.04.  

```shell

## env for pruning attack
conda env create -f ./envs/pruning_attack.yaml
cp torch/torch_pruning_attack/torch/*  ~/anaconda2/envs/pruning/lib/python3.5/site-packages/torch/ -r

## env for quantization attack
conda env create -f ./envs/quantization_attack.yaml
cp torch/torch_quantization_attack/torch/*  ~/anaconda2/envs/quantization/lib/python3.7/site-packages/torch/ -r

## env for testing
conda env create -f ./envs/clean_env.yaml 

```

## Attacks for Model Pruning

Train clean models, and extract layer-level pruning rates.
```shell
## train clean models
cd train_baseline_models 
./scripts/clean_model_training.sh

## extract pruning rates
./scripts/get_layer_level_pruning_rate.sh
```

Train and test backdoored models. Results will be stored in _./train_pruning_attack/pruning_results/_ folder.

```shell
## standard known rate attack
cd ../train_pruning_attack/
./scripts/known_rate_attack.sh  

## distilled known rate attack 
./scripts/known_rate_attack_distilled.sh 

## standard unknown rate attack
./scripts/unknown_rate_attack.sh

## distilled unknown rate attack
./scripts/unknown_rate_attack_distilled.sh

```

## Attacks for Model Quantization


Train backdoored models.

```shell
## standard attack
cd ../train_quantization_attack/
./scripts/attack_standard.sh  

## distilled attack 
./scripts/attack_distilled.sh 

```

Evaluate attack effectiveness when the deployer quantizes the model using different calibration datasets.
```shell
## standard attack;  same distribution
./scripts/eval/eval_same_distribution.sh  

## standard attack;  similar distribution
./scripts/eval/eval_similar_distribution.sh

## standard attack;  dissimilar distribution
./scripts/eval/eval_dissimilar_distribution.sh  

## distilled attack;  same distribution
./scripts/eval/eval_same_distribution_distilled.sh  

## distilled attack;  similar distribution
./scripts/eval/eval_similar_distribution_distilled.sh  

## distilled attack;  dissimilar distribution
./scripts/eval/eval_dissimilar_distribution_distilled.sh 
```



## Trojan Detection -- MNTD


Train baseline clean models for comparison

```shell
## train clean models
cd ../train_baseline_models 
./scripts/clean_model_training_for_detection.sh
```
MNTD detection
```shell
## train shadow models and meta classifiers
cd ../trojan_detection_mntd/
./train.sh

## MNTD evaluation
./run.sh
```



## Trojan Detection -- Neural Cleanse


Train baseline clean models for comparison (This step can be skipped if the MNTD part is finished.)

```shell
## train clean models (This step can be skipped if the MNTD part is finished.)
cd ../train_baseline_models 
./scripts/clean_model_training_for_detection.sh
```

Neural Cleanse detection
```shell
## get anomaly indexes
cd ../trojan_detection_neural_cleanse/
./scripts/neural_cleanse_known_rate.sh              
./scripts/neural_cleanse_known_rate_distilled.sh         
./scripts/neural_cleanse_unknown_rate.sh
./scripts/neural_cleanse_unknown_rate_distilled.sh
./scripts/neural_cleanse_quantization.sh
./scripts/neural_cleanse_quantization_distilled.sh  
./scripts/neural_cleanse_baseline_clean_model.sh  
./scripts/neural_cleanse_get_anomaly_index.sh

## calculate AUC
./scripts/neural_cleanse_cal_auc.sh
## calculate AUC, only consider anomaly indexes of the target classes
./scripts/neural_cleanse_cal_auc_diagonal.sh
```
