#!/bin/bash

cuda_device=7

dataset='cifar10'

source ~/anaconda2/bin/activate clean
export CUDA_VISIBLE_DEVICES=$cuda_device


pushd .



##################### scripts for clean baseline clean models

# python mad_outlier_detection_batch_heat_map_v4.py --full --clean  --dataset gtsrb  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4.py --full --clean  --dataset gtsrb  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4.py --full --clean  --dataset gtsrb  --network-arch mobilenet
# python mad_outlier_detection_batch_heat_map_v4.py --full --clean  --dataset cifar10  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4.py --full --clean  --dataset cifar10  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4.py --full --clean  --dataset cifar10  --network-arch mobilenet

####################  scripts for real backdoored models

# python mad_outlier_detection_batch_heat_map_v4.py --full --dataset gtsrb  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4.py --full --dataset gtsrb  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4.py --full --dataset gtsrb  --network-arch mobilenet
# python mad_outlier_detection_batch_heat_map_v4.py --full --dataset cifar10  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4.py --full --dataset cifar10  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4.py --full --dataset cifar10  --network-arch mobilenet

#################### scripts for pruning attacks targeting known pruning rates

# python mad_outlier_detection_batch_heat_map_v4_last_upgrade.py  --try-version ZZZZC2lastupgrade   --save-version 4 --full --dataset gtsrb  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4_last_upgrade.py  --try-version ZZZZC2lastupgrade   --save-version 4 --full --dataset gtsrb  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4_last_upgrade.py  --try-version ZZZZC2lastupgrade   --save-version 4 --full --dataset gtsrb  --network-arch mobilenet
# python mad_outlier_detection_batch_heat_map_v4_last_upgrade.py  --try-version ZZZZC2lastupgrade   --save-version 4 --full --dataset cifar10  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4_last_upgrade.py  --try-version ZZZZC2lastupgrade   --save-version 4 --full --dataset cifar10  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4_last_upgrade.py  --try-version ZZZZC2lastupgrade   --save-version 4 --full --dataset cifar10  --network-arch mobilenet

# python mad_outlier_detection_batch_heat_map_v4_last_upgrade.py  --try-version ZZZZC2lastupgradeteacher   --save-version 4 --full --dataset gtsrb  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4_last_upgrade.py  --try-version ZZZZC2lastupgradeteacher   --save-version 4 --full --dataset gtsrb  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4_last_upgrade.py  --try-version ZZZZC2lastupgradeteacher   --save-version 4 --full --dataset gtsrb  --network-arch mobilenet
# python mad_outlier_detection_batch_heat_map_v4_last_upgrade.py  --try-version ZZZZC2lastupgradeteacher   --save-version 4 --full --dataset cifar10  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4_last_upgrade.py  --try-version ZZZZC2lastupgradeteacher   --save-version 4 --full --dataset cifar10  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4_last_upgrade.py  --try-version ZZZZC2lastupgradeteacher   --save-version 4 --full --dataset cifar10  --network-arch mobilenet

# python mad_outlier_detection_batch_heat_map_v4_last_upgradeclean.py  --try-version ZZZZC2lastupgradeclean   --save-version 4 --full --dataset gtsrb  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4_last_upgradeclean.py  --try-version ZZZZC2lastupgradeclean   --save-version 4 --full --dataset gtsrb  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4_last_upgradeclean.py  --try-version ZZZZC2lastupgradeclean   --save-version 4 --full --dataset gtsrb  --network-arch mobilenet
# python mad_outlier_detection_batch_heat_map_v4_last_upgradeclean.py  --try-version ZZZZC2lastupgradeclean   --save-version 4 --full --dataset cifar10  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4_last_upgradeclean.py  --try-version ZZZZC2lastupgradeclean   --save-version 4 --full --dataset cifar10  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4_last_upgradeclean.py  --try-version ZZZZC2lastupgradeclean   --save-version 4 --full --dataset cifar10  --network-arch mobilenet


# python mad_outlier_detection_batch_heat_map_v4_last_upgrade_compressed.py  --try-version ZZZZC2lastupgrade   --save-version 4 --full --dataset cifar10  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4_last_upgrade_compressed.py  --try-version ZZZZC2lastupgrade   --save-version 4 --full --dataset cifar10  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4_last_upgrade_compressed.py  --try-version ZZZZC2lastupgrade   --save-version 4 --full --dataset cifar10  --network-arch mobilenet

# python mad_outlier_detection_batch_heat_map_v4_last_upgrade_compressed.py  --try-version ZZZZC2lastupgradeteacher   --save-version 4 --full --dataset cifar10  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4_last_upgrade_compressed.py  --try-version ZZZZC2lastupgradeteacher   --save-version 4 --full --dataset cifar10  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4_last_upgrade_compressed.py  --try-version ZZZZC2lastupgradeteacher   --save-version 4 --full --dataset cifar10  --network-arch mobilenet

# python mad_outlier_detection_batch_heat_map_v4_2u_compressed.py  --try-version ZZZZC2uteacher   --save-version 4 --full --dataset cifar10  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4_2u_compressed.py  --try-version ZZZZC2uteacher   --save-version 4 --full --dataset cifar10  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4_2u_compressed.py  --try-version ZZZZC2uteacher   --save-version 4 --full --dataset cifar10  --network-arch mobilenet



#################### scripts for pruning attacks targeting unknown pruning rates

# python mad_outlier_detection_batch_heat_map_v4_2u.py  --try-version ZZZZC2u   --save-version 4 --full --dataset gtsrb  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4_2u.py  --try-version ZZZZC2u   --save-version 4 --full --dataset gtsrb  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4_2u.py  --try-version ZZZZC2u   --save-version 4 --full --dataset gtsrb  --network-arch mobilenet
# python mad_outlier_detection_batch_heat_map_v4_2u.py  --try-version ZZZZC2u   --save-version 4 --full --dataset cifar10  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4_2u.py  --try-version ZZZZC2u   --save-version 4 --full --dataset cifar10  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4_2u.py  --try-version ZZZZC2u   --save-version 4 --full --dataset cifar10  --network-arch mobilenet

# python mad_outlier_detection_batch_heat_map_v4_2u.py  --try-version ZZZZC2uteacher   --save-version 4 --full --dataset gtsrb  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4_2u.py  --try-version ZZZZC2uteacher   --save-version 4 --full --dataset gtsrb  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4_2u.py  --try-version ZZZZC2uteacher   --save-version 4 --full --dataset gtsrb  --network-arch mobilenet
# python mad_outlier_detection_batch_heat_map_v4_2u.py  --try-version ZZZZC2uteacher   --save-version 4 --full --dataset cifar10  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4_2u.py  --try-version ZZZZC2uteacher   --save-version 4 --full --dataset cifar10  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4_2u.py  --try-version ZZZZC2uteacher   --save-version 4 --full --dataset cifar10  --network-arch mobilenet

# python mad_outlier_detection_batch_heat_map_v4_2uclean.py  --try-version ZZZZC2uclean   --save-version 4 --full --dataset gtsrb  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4_2uclean.py  --try-version ZZZZC2uclean   --save-version 4 --full --dataset gtsrb  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4_2uclean.py  --try-version ZZZZC2uclean   --save-version 4 --full --dataset gtsrb  --network-arch mobilenet
# python mad_outlier_detection_batch_heat_map_v4_2uclean.py  --try-version ZZZZC2uclean   --save-version 4 --full --dataset cifar10  --network-arch vgg
# python mad_outlier_detection_batch_heat_map_v4_2uclean.py  --try-version ZZZZC2uclean   --save-version 4 --full --dataset cifar10  --network-arch resnet18
# python mad_outlier_detection_batch_heat_map_v4_2uclean.py  --try-version ZZZZC2uclean   --save-version 4 --full --dataset cifar10  --network-arch mobilenet


#################### scripts for quantizaiton attack

# python mad_outlier_detection_batch_heat_map_v4_quantization.py  --dataset gtsrb  --version 1026 --network-arch vgg  --max-epoch 35 --full
# python mad_outlier_detection_batch_heat_map_v4_quantization.py  --dataset gtsrb  --version 1026 --network-arch resnet18  --max-epoch 35 --full
# python mad_outlier_detection_batch_heat_map_v4_quantization.py  --dataset gtsrb  --version 1026 --network-arch mobilenet  --max-epoch 35 --full
# python mad_outlier_detection_batch_heat_map_v4_quantization.py  --dataset cifar10  --version 1026 --network-arch vgg  --max-epoch 35 --full
# python mad_outlier_detection_batch_heat_map_v4_quantization.py  --dataset cifar10  --version 1026 --network-arch resnet18  --max-epoch 35 --full
# python mad_outlier_detection_batch_heat_map_v4_quantization.py  --dataset cifar10  --version 1026 --network-arch mobilenet  --max-epoch 35 --full



# python mad_outlier_detection_batch_heat_map_v4_quantization.py  --dataset gtsrb  --version 1028 --network-arch vgg  --max-epoch 15 --full
# python mad_outlier_detection_batch_heat_map_v4_quantization.py  --dataset gtsrb  --version 1028 --network-arch resnet18  --max-epoch 15 --full
# python mad_outlier_detection_batch_heat_map_v4_quantization.py  --dataset gtsrb  --version 1028 --network-arch mobilenet  --max-epoch 15 --full

# python mad_outlier_detection_batch_heat_map_v4_quantization.py  --dataset cifar10  --version 1028 --network-arch vgg  --max-epoch 15 --full
# python mad_outlier_detection_batch_heat_map_v4_quantization.py  --dataset cifar10  --version 1028 --network-arch resnet18  --max-epoch 15 --full
# python mad_outlier_detection_batch_heat_map_v4_quantization.py  --dataset cifar10  --version 1028 --network-arch mobilenet  --max-epoch 15 --full


# CUDA_VISIBLE_DEVICES=7 python verify_models_average.py --network vgg  --max_epoch 15  --margin 0.5   --dataset gtsrb  --version 1028
# CUDA_VISIBLE_DEVICES=7 python verify_models_average.py --network resnet18  --max_epoch 15  --margin 0.5   --dataset gtsrb  --version 1028
# CUDA_VISIBLE_DEVICES=7 python verify_models_average.py --network mobilenet  --max_epoch 15  --margin 0.5   --dataset gtsrb  --version 1028



# python mad_outlier_detection_batch_heat_map_v4_quantizationclean.py  --dataset gtsrb  --version 10011 --network-arch vgg  --max-epoch 35 --full
# python mad_outlier_detection_batch_heat_map_v4_quantizationclean.py  --dataset gtsrb  --version 10011 --network-arch resnet18  --max-epoch 35 --full
# python mad_outlier_detection_batch_heat_map_v4_quantizationclean.py  --dataset gtsrb  --version 10011 --network-arch mobilenet  --max-epoch 35 --full

# python mad_outlier_detection_batch_heat_map_v4_quantizationclean.py  --dataset cifar10  --version 10011 --network-arch vgg  --max-epoch 35 --full
# python mad_outlier_detection_batch_heat_map_v4_quantizationclean.py  --dataset cifar10  --version 10011 --network-arch resnet18  --max-epoch 35 --full
# python mad_outlier_detection_batch_heat_map_v4_quantizationclean.py  --dataset cifar10  --version 10011 --network-arch mobilenet  --max-epoch 35 --full


# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch vgg  --try-version trigger
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch resnet18 --try-version trigger
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch mobilenet --try-version trigger

# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch vgg --try-version trigger
# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch resnet18 --try-version trigger
# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch mobilenet --try-version trigger


# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch vgg  --try-version ZZZZC2lastupgrade
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch resnet18 --try-version ZZZZC2lastupgrade
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch mobilenet --try-version ZZZZC2lastupgrade

# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch vgg --try-version ZZZZC2lastupgrade
# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch resnet18 --try-version ZZZZC2lastupgrade
# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch mobilenet --try-version ZZZZC2lastupgrade


# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch vgg  --try-version ZZZZC2lastupgradeteacher
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch resnet18 --try-version ZZZZC2lastupgradeteacher
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch mobilenet --try-version ZZZZC2lastupgradeteacher

# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch vgg --try-version ZZZZC2lastupgradeteacher
# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch resnet18 --try-version ZZZZC2lastupgradeteacher
# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch mobilenet --try-version ZZZZC2lastupgradeteacher


# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch vgg  --try-version ZZZZC2u
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch resnet18 --try-version ZZZZC2u
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch mobilenet --try-version ZZZZC2u

# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch vgg --try-version ZZZZC2u
# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch resnet18 --try-version ZZZZC2u
# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch mobilenet --try-version ZZZZC2u


# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch vgg  --try-version ZZZZC2uteacher
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch resnet18 --try-version ZZZZC2uteacher
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch mobilenet --try-version ZZZZC2uteacher

# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch vgg --try-version ZZZZC2uteacher
# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch resnet18 --try-version ZZZZC2uteacher
# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch mobilenet --try-version ZZZZC2uteacher


# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch vgg  --try-version quantization1026
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch resnet18 --try-version quantization1026
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch mobilenet --try-version quantization1026

# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch vgg --try-version quantization1026
# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch resnet18 --try-version quantization1026
# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch mobilenet --try-version quantization1026


# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch vgg  --try-version quantization1028
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch resnet18 --try-version quantization1028
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch mobilenet --try-version quantization1028

# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch vgg --try-version quantization1028
# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch resnet18 --try-version quantization1028
# python mad_calculate_auc_diagonal.py --dataset gtsrb  --network-arch mobilenet --try-version quantization1028

#####################

# python mad_calculate_auc.py --dataset cifar10 --network-arch vgg  --try-version trigger
# python mad_calculate_auc.py --dataset cifar10 --network-arch resnet18 --try-version trigger
# python mad_calculate_auc.py --dataset cifar10 --network-arch mobilenet --try-version trigger

# python mad_calculate_auc.py --dataset gtsrb  --network-arch vgg --try-version trigger
# python mad_calculate_auc.py --dataset gtsrb  --network-arch resnet18 --try-version trigger
# python mad_calculate_auc.py --dataset gtsrb  --network-arch mobilenet --try-version trigger


# python mad_calculate_auc.py --dataset cifar10 --network-arch vgg  --try-version ZZZZC2lastupgrade
# python mad_calculate_auc.py --dataset cifar10 --network-arch resnet18 --try-version ZZZZC2lastupgrade
# python mad_calculate_auc.py --dataset cifar10 --network-arch mobilenet --try-version ZZZZC2lastupgrade


# python mad_calculate_auc.py --dataset cifar10 --network-arch vgg  --try-version ZZZZC2lastupgrade
# python mad_calculate_auc.py --dataset cifar10 --network-arch resnet18 --try-version ZZZZC2lastupgrade
# python mad_calculate_auc.py --dataset cifar10 --network-arch mobilenet --try-version ZZZZC2lastupgrade


# python mad_calculate_auc.py --dataset gtsrb  --network-arch vgg --try-version ZZZZC2lastupgrade
# python mad_calculate_auc.py --dataset gtsrb  --network-arch resnet18 --try-version ZZZZC2lastupgrade
# python mad_calculate_auc.py --dataset gtsrb  --network-arch mobilenet --try-version ZZZZC2lastupgrade


# python mad_calculate_auc.py --dataset cifar10 --network-arch vgg  --try-version ZZZZC2lastupgradeteacher
# python mad_calculate_auc.py --dataset cifar10 --network-arch resnet18 --try-version ZZZZC2lastupgradeteacher
# python mad_calculate_auc.py --dataset cifar10 --network-arch mobilenet --try-version ZZZZC2lastupgradeteacher

# python mad_calculate_auc.py --dataset gtsrb  --network-arch vgg --try-version ZZZZC2lastupgradeteacher
# python mad_calculate_auc.py --dataset gtsrb  --network-arch resnet18 --try-version ZZZZC2lastupgradeteacher
# python mad_calculate_auc.py --dataset gtsrb  --network-arch mobilenet --try-version ZZZZC2lastupgradeteacher


# python mad_calculate_auc.py --dataset cifar10 --network-arch vgg  --try-version ZZZZC2lastupgradeteachercompressed
# python mad_calculate_auc.py --dataset cifar10 --network-arch resnet18 --try-version ZZZZC2lastupgradeteachercompressed
# python mad_calculate_auc.py --dataset cifar10 --network-arch mobilenet --try-version ZZZZC2lastupgradeteachercompressed


# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch vgg  --try-version ZZZZC2lastupgradeteachercompressed
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch resnet18 --try-version ZZZZC2lastupgradeteachercompressed
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch mobilenet --try-version ZZZZC2lastupgradeteachercompressed



python mad_calculate_auc.py --dataset cifar10 --network-arch vgg  --try-version ZZZZC2uteachercompressed
# python mad_calculate_auc.py --dataset cifar10 --network-arch resnet18 --try-version ZZZZC2uteachercompressed
# python mad_calculate_auc.py --dataset cifar10 --network-arch mobilenet --try-version ZZZZC2uteachercompressed


python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch vgg  --try-version ZZZZC2uteachercompressed
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch resnet18 --try-version ZZZZC2uteachercompressed
# python mad_calculate_auc_diagonal.py --dataset cifar10 --network-arch mobilenet --try-version ZZZZC2uteachercompressed





# python mad_calculate_auc.py --dataset cifar10 --network-arch vgg  --try-version ZZZZC2u
# python mad_calculate_auc.py --dataset cifar10 --network-arch resnet18 --try-version ZZZZC2u
# python mad_calculate_auc.py --dataset cifar10 --network-arch mobilenet --try-version ZZZZC2u

# python mad_calculate_auc.py --dataset gtsrb  --network-arch vgg --try-version ZZZZC2u
# python mad_calculate_auc.py --dataset gtsrb  --network-arch resnet18 --try-version ZZZZC2u
# python mad_calculate_auc.py --dataset gtsrb  --network-arch mobilenet --try-version ZZZZC2u


# python mad_calculate_auc.py --dataset cifar10 --network-arch vgg  --try-version ZZZZC2uteacher
# python mad_calculate_auc.py --dataset cifar10 --network-arch resnet18 --try-version ZZZZC2uteacher
# python mad_calculate_auc.py --dataset cifar10 --network-arch mobilenet --try-version ZZZZC2uteacher

# python mad_calculate_auc.py --dataset gtsrb  --network-arch vgg --try-version ZZZZC2uteacher
# python mad_calculate_auc.py --dataset gtsrb  --network-arch resnet18 --try-version ZZZZC2uteacher
# python mad_calculate_auc.py --dataset gtsrb  --network-arch mobilenet --try-version ZZZZC2uteacher

# python mad_calculate_auc.py --dataset cifar10 --network-arch vgg  --try-version quantization1026
# python mad_calculate_auc.py --dataset cifar10 --network-arch resnet18 --try-version quantization1026
# python mad_calculate_auc.py --dataset cifar10 --network-arch mobilenet --try-version quantization1026

# python mad_calculate_auc.py --dataset gtsrb  --network-arch vgg --try-version quantization1026
# python mad_calculate_auc.py --dataset gtsrb  --network-arch resnet18 --try-version quantization1026
# python mad_calculate_auc.py --dataset gtsrb  --network-arch mobilenet --try-version quantization1026


# python mad_calculate_auc.py --dataset cifar10 --network-arch vgg  --try-version quantization1028
# python mad_calculate_auc.py --dataset cifar10 --network-arch resnet18 --try-version quantization1028
# python mad_calculate_auc.py --dataset cifar10 --network-arch mobilenet --try-version quantization1028

# python mad_calculate_auc.py --dataset gtsrb  --network-arch vgg --try-version quantization1028
# python mad_calculate_auc.py --dataset gtsrb  --network-arch resnet18 --try-version quantization1028
# python mad_calculate_auc.py --dataset gtsrb  --network-arch mobilenet --try-version quantization1028



###############



# python mad_calculate_auc.py --dataset cifar10 --network-arch vgg  --try-version ZZZZC2lastupgradeclean
# python mad_calculate_auc.py --dataset cifar10 --network-arch resnet18 --try-version ZZZZC2lastupgradeclean
# python mad_calculate_auc.py --dataset cifar10 --network-arch mobilenet --try-version ZZZZC2lastupgradeclean

# python mad_calculate_auc.py --dataset gtsrb  --network-arch vgg --try-version ZZZZC2lastupgradeclean
# python mad_calculate_auc.py --dataset gtsrb  --network-arch resnet18 --try-version ZZZZC2lastupgradeclean
# python mad_calculate_auc.py --dataset gtsrb  --network-arch mobilenet --try-version ZZZZC2lastupgradeclean

# python mad_calculate_auc.py --dataset cifar10 --network-arch vgg  --try-version ZZZZC2uclean
# python mad_calculate_auc.py --dataset cifar10 --network-arch resnet18 --try-version ZZZZC2uclean
# python mad_calculate_auc.py --dataset cifar10 --network-arch mobilenet --try-version ZZZZC2uclean

# python mad_calculate_auc.py --dataset gtsrb  --network-arch vgg --try-version ZZZZC2uclean
# python mad_calculate_auc.py --dataset gtsrb  --network-arch resnet18 --try-version ZZZZC2uclean
# python mad_calculate_auc.py --dataset gtsrb  --network-arch mobilenet --try-version ZZZZC2uclean



# python mad_calculate_auc.py --dataset cifar10 --network-arch vgg  --try-version quantization10011
# python mad_calculate_auc.py --dataset cifar10 --network-arch resnet18 --try-version quantization10011
# python mad_calculate_auc.py --dataset cifar10 --network-arch mobilenet --try-version quantization10011

# python mad_calculate_auc.py --dataset gtsrb  --network-arch vgg --try-version quantization10011
# python mad_calculate_auc.py --dataset gtsrb  --network-arch resnet18 --try-version quantization10011
# python mad_calculate_auc.py --dataset gtsrb  --network-arch mobilenet --try-version quantization10011



popd