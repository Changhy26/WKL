gpu=0

# For ImageNet
#CUDA_VISIBLE_DEVICES=$gpu python tools/generate_class_proto.py --model ResNet34
#CUDA_VISIBLE_DEVICES=$gpu python tools/generate_cost_matrix.py --model ResNet34

# CUDA_VISIBLE_DEVICES=$gpu python tools/generate_class_proto.py --model ResNet50
# CUDA_VISIBLE_DEVICES=$gpu python tools/generate_cost_matrix.py --model ResNet50

# For CIFAR-100
# CUDA_VISIBLE_DEVICES=$gpu python tools/generate_class_proto.py --model resnet32x4 --dataset cifar100 --ckpt download_ckpts/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth --feature_type feature
# CUDA_VISIBLE_DEVICES=$gpu python tools/generate_cost_matrix.py --model resnet32x4 --dataset cifar100

# CUDA_VISIBLE_DEVICES=$gpu python tools/generate_class_proto.py --model wrn_40_2 --dataset cifar100 --ckpt download_ckpts/cifar_teachers/wrn_40_2_vanilla/ckpt_epoch_240.pth --feature_type feature
# CUDA_VISIBLE_DEVICES=$gpu python tools/generate_cost_matrix.py --model wrn_40_2 --dataset cifar100

# ResNet50
CUDA_VISIBLE_DEVICES=$gpu python tools/generate_class_proto.py --model ResNet50 --dataset cifar100 --ckpt download_ckpts/cifar_teachers/ResNet50_vanilla/ckpt_epoch_240.pth --feature_type feature
CUDA_VISIBLE_DEVICES=$gpu python tools/generate_cost_matrix.py --model ResNet50 --dataset cifar100

# ResNet56
#CUDA_VISIBLE_DEVICES=$gpu python tools/generate_class_proto.py --model resnet56 --dataset cifar100 --ckpt download_ckpts/cifar_teachers/resnet56_vanilla/ckpt_epoch_240.pth --feature_type feature
#CUDA_VISIBLE_DEVICES=$gpu python tools/generate_cost_matrix.py --model resnet56 --dataset cifar100

# ResNet110
#CUDA_VISIBLE_DEVICES=$gpu python tools/generate_class_proto.py --model resnet110 --dataset cifar100 --ckpt download_ckpts/cifar_teachers/resnet110_vanilla/ckpt_epoch_240.pth --feature_type feature
#CUDA_VISIBLE_DEVICES=$gpu python tools/generate_cost_matrix.py --model resnet110 --dataset cifar100

# vgg13
#CUDA_VISIBLE_DEVICES=$gpu python tools/generate_class_proto.py --model resnet110 --dataset cifar100 --ckpt download_ckpts/cifar_teachers/resnet110_vanilla/ckpt_epoch_240.pth --feature_type feature
#CUDA_VISIBLE_DEVICES=$gpu python tools/generate_cost_matrix.py --model resnet110 --dataset cifar100