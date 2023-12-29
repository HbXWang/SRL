# Structure-dirven representation learning for deep clustering (SRL)

This is our PyTorch implementation for the paper:
Xiang Wang, Huafeng Liu, Liping Jing Jian Yu. Structure-driven Representation Learning for Deep Clustering. ACM Transactions on Knowledge Discovery from Data (TKDD), 2023.


# Dependency

- python>=3.7
- pytorch>=1.6.0
- torchvision>=0.7.0
- munkres>=1.0.7
- numpy>=1.20.1
- opencv-python>=4.5.2.52
- pyyaml>=5.3.1
- scikit-learn>=0.24.2
- tqdm>=4.60.0

# Usage

## Configuration

There is a configuration file "config/config.yaml", where one can edit both the training and test options.

## Training

After setting the configuration, to start training, simply run

> python train.py

## Test

Once the training is completed, there will be a saved model in the "model_save_dir" specified in the configuration file. To test the trained model, edit configuration file and run

> python evaluate.py

# Dataset

CIFAR-10, CIFAR-100, STL-10 will be automatically downloaded by Pytorch. For ImageNet-10 and ImageNet-dogs, we provided their description in the "dataset" folder.
