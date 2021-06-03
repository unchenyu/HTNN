import sys
import os
import os.path as pth

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
import argparse

import nets
import datasets
import tools
import layers as L

model_names = ['resnet20', 'vggnaga', 'cnnc']
dataset_names = ['cifar10', 'cifar100']
parser = argparse.ArgumentParser(description='PyTorch HTNN Evaluation')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture (default: resnet20)')
parser.add_argument('--dataset', metavar='DATA', default='cifar10',
                    choices=dataset_names,
                    help='dataset (default: cifar10')

# Calculate weight density
def cal_density(model):
    num_pruned, num_weights = 0, 0
    for m in model.modules():
        if isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
            num = torch.numel(m.weight.data)
            weight_mask = (abs(m.weight.data) > 0).float()

            num_pruned += num - torch.sum(weight_mask)
            num_weights += num

    return 1 - num_pruned / num_weights

def eval(model, testloader):
    print("Running evaluation on validation split")
    model.eval()

    lossfunc = nn.CrossEntropyLoss().cuda()
    error_top1 = []
    error_top5 = []
    vld_loss = []
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            error_top1.append(tools.topK_error(outputs, labels, K=1).item())
            error_top5.append(tools.topK_error(outputs, labels, K=5).item())
            vld_loss.append(lossfunc(outputs, labels).item())

        error_top1 = np.average(error_top1)
        error_top5 = np.average(error_top5)
        vld_loss = np.average(vld_loss)
        print("-- Validation result -- acc_top1: %.4f acc_top5: %.4f loss:%.4f" % (1-error_top1, 1-error_top5, vld_loss))

def main():
    args = parser.parse_args()

    if args.arch == 'resnet20':
        model = nets.RESNET20MULT3()
        pretrained_checkpoint = '../checkpoints/resnet20_ckpt.tar'
    if args.arch == 'vggnaga':
        model = nets.VGGnagaMULT()
        pretrained_checkpoint = '../checkpoints/vggnaga_ckpt.tar'
    if args.arch == 'cnnc':
        model = nets.CNNCMULT3()
        pretrained_checkpoint = '../checkpoints/cnnc_ckpt.tar'
    model.cuda()

    # load pretrained checkpoint
    print("Loading checkpoint '{}'".format(pretrained_checkpoint))
    pretrained_ckpt = torch.load(pretrained_checkpoint)
    model.load_state_dict(pretrained_ckpt['state_dict'])
    print("Loaded checkpoint '{}'".format(pretrained_checkpoint))

    density = cal_density(model)
    print("-- Weight desity after learning sparsity and CSD quantization: %.4f" % (density))

    if args.dataset == 'cifar10':
        _, testloader = datasets.get_cifar10()
    if args.dataset == 'cifar100':
        _, testloader = datasets.get_cifar100()

    eval(model, testloader)


if __name__ == '__main__':
    main()
