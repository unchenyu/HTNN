# HTNN: Deep Learning in Heterogeneous Transform Domains with Sparse-Orthogonal Weights

This repository is the code of paper *HTNN: Deep Learning in Heterogeneous Transform Domains with Sparse-Orthogonal Weights*.

# Introduction
We present a new class of transform domain deep neural networks (DNNs), where convolution operations are replaced by element-wise multiplications in heterogeneous transform domains. To further reduce the network complexity, we propose a framework to learn sparse-orthogonal weights in heterogeneous transform domains co-optimized with a hardware-efficient accelerator architecture to minimize the overhead of handling sparse weights. Furthermore, sparse-orthogonal weights are non-uniformly quantized with canonical-signed-digit (CSD) representations to substitute multiplications with simpler additions. The proposed approach reduces the complexity by a factor of 4.9 -- 6.8$\times$ without compromising the DNN accuracy compared to equivalent CNNs that employ sparse (pruned) weights.
![HTNN layer](https://github.com/unchenyu/HTNN/blob/main/images/HTNN_layer.png)

## Datasets
We use CIFAR-10 & CIFAR-100 datasets to train and evaluate our models. You can use our scripts to download them automatically or manually donload [here](https://www.cs.toronto.edu/~kriz/cifar.html).

# Code

## Requirements

1. Python 3.6 or higher.

2. torch >= 1.4 and torchvision >= 0.5. Follow the instruction on [official PyTorch website](https://pytorch.org/get-started/locally/) to install.

3. GPU support with CUDA 10.1 or higher.

## Train the models

You can use our code package to train HTNN with ResNet-20, VGG-nagadomi and ConvPool-CNN-C architectures. For example, train HTNN model with ResNet-20 on CIFAR-10:

```sh
$ python train.py --arch='resnet20' --dataset='cifar10'
```

Apply structured ADMM pruning to get sparse-orthogonal weights:

```sh
$ python prune.py --arch='resnet20' --dataset='cifar10'
```

Apply CSD quantization to get quantized weights:

```sh
$ python quant.py --arch='resnet20' --dataset='cifar10'
```

## Evaluate pretrained models

We provide 3 pretrained models for HTNN wtih ResNet-20, VGG-nagadomi and ConvPool-CNN-C architectures respectively.

To evaluate HTNN with ResNet-20 on CIFAR-10 dataset, run:

```sh
$ python3 eval.py --arch='resnet20' --dataset='cifar10'
```

To evaluate HTNN with VGG-nagadomi on CIFAR-10 dataset, run:

```sh
$ python3 eval.py --arch='vggnaga' --dataset='cifar10'
```

To evaluate HTNN with ConvPool-CNN-C on CIFAR-100 dataset, run:

```sh
$ python3 eval.py --arch='cnnc' --dataset='cifar100'
```

## Results

Our models achieve the following performance:


| Model name          | Top 1 Accuracy  | Weight Desity  |  # of OPs  | Complexity reduction to sparse CNN |
|:-------------------:|:---------------:|:--------------:|:----------:|:----------------------------------:|
| HTNN ResNet-20      |     91.56%      |      24.5%     |   21.2M    |            6.2X                    |
| HTNN VGG-nagadomi   |     93.01%      |      34.9%     |   78.2M    |            4.9X                    |
| HTNN ConvPool-CNN-C |     70.51%      |      19.0%     |   88.3M    |            6.8X                    |

# Citation
Please cite our paper if you find our paper useful for your research.