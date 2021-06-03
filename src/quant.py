import sys
import os
import os.path as pth
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

import nets
import datasets
import tools
import layers as L


# =====================================
# Training configuration default params
# =====================================
parser = argparse.ArgumentParser(description='PyTorch HTNN Training')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size (default: 256)')
parser.add_argument('--admm_epochs', type=int, default=30,
                    help='Number of ADMM pruning epochs (default: 30')
parser.add_argument('--retraining_epochs', type=int, default=30,
                    help='Number of retraining epochs (default: 30')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='Base learning rate (default: 0.001')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay rate (default: 0.0005')
parser.add_argument('--pretrained_weights', type=str, default='../example/ckpt_prune/checkpoint_180.tar',
                    help='Path to pretrained model')
parser.add_argument('--checkpoint_dir', type=str, default='../example/ckpt_quant',
                    help='Folder to save checkpoints')
parser.add_argument('--log_path', type=str, default='../example/quant.log',
                    help='Path to write logs into')
model_names = ['resnet20', 'vggnaga', 'cnnc']
dataset_names = ['cifar10', 'cifar100']
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture (default: resnet20)')
parser.add_argument('--dataset', metavar='DATA', default='cifar10',
                    choices=dataset_names,
                    help='dataset (default: cifar10')


# Retrive pruning masks
def retrive_masks(model):
    num_pruned, num_weights = 0, 0
    weight_masks = []
    for m in model.modules():
        if isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
            num = torch.numel(m.weight.data)
            weight_mask = (abs(m.weight.data) > 0).float()
            weight_masks.append(weight_mask)

            num_pruned += num - torch.sum(weight_mask)
            num_weights += num

    print('-- compress rate: %.4f' % (num_pruned / num_weights))
    return weight_masks

def apply_mask(model, weight_masks):
    idx = 0
    for m in model.modules():
        if isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
            m.weight.data *= weight_masks[idx]
            idx += 1

import itertools
def getQlevels(bits=6):
    qlevels = [0]
    itertable = []
    for i in range(bits-1):
        qlevels.extend((2**i, -2**i))
        itertable.extend((2**i, -2**i))
    itertable.extend((2**(bits-1), -2**(bits-1)))
    comb2 = list(itertools.combinations(itertable, 2))
    for item in comb2:
        val = item[0] + item[1]
        if val < 2**(bits-1) and val >  -(2**(bits-1)):
            qlevels.append(val)
    qlevels = sorted(list(dict.fromkeys(qlevels)))
    return qlevels

def projection(weights):
    weight = weights.cpu().numpy().flatten()

    alpha = max(abs(weight))/max(_QLEVELS)
    alpha = 2**(np.round(np.log2(alpha)))
    # iterative solver for scaling factor alpha
    for i in range(5):
        weight_int = np.round(weight/alpha)  
        alpha = np.sum(weight*weight_int)/np.sum(weight_int*weight_int)
        alpha = 2**(np.round(np.log2(alpha)))

    # print("alpha: {:.4f}".format(alpha))
    weight_int = np.zeros(weight.shape)
    for i, val in enumerate(weight):
        if val != 0:
            abs_diff_func = lambda _QLEVELS : abs(_QLEVELS - val/alpha)
            weight_int[i] = min(_QLEVELS, key=abs_diff_func)

    weight_quant = torch.tensor(weight_int.astype(np.float32()) * alpha)
    return weight_quant.reshape(weights.shape)

def quantize(weights, bound):
    weight = weights.cpu().numpy().flatten()
    weight[weight>bound] = bound
    weight[weight<-bound] = -bound
    alpha = bound/max(_QLEVELS)
    alpha = 2**(np.round(np.log2(alpha)))

    # print("alpha: {:.4f}".format(alpha))
    weight_int = np.zeros(weight.shape)
    for i, val in enumerate(weight):
        if val != 0:
            abs_diff_func = lambda _QLEVELS : abs(_QLEVELS - val/alpha)
            weight_int[i] = min(_QLEVELS, key=abs_diff_func)

    weight_quant = torch.tensor(weight_int.astype(np.float32()) * alpha)
    return weight_quant.reshape(weights.shape).cuda()

def admm_loss(model, base_lossfunc, Z, U, output, target):
    idx = 0
    loss = base_lossfunc(output, target)
    for m in model.modules():
        if isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
            u = U[idx].cuda()
            z = Z[idx].cuda()
            loss += 1e-4 / 2 * (m.weight.data - z + u).norm()
            idx += 1
    return loss

def initialize_Z_and_U(model):
    Z = ()
    U = ()
    for m in model.modules():
        if isinstance(m, L.MultLayer)or isinstance(m, L.MultLayer3):
            Z += (m.weight.data.cpu().clone(),)
            U += (torch.zeros_like(m.weight.data).cpu(),)
    return Z, U

def update_X(model):
    X = ()
    for m in model.modules():
        if isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
            X += (m.weight.data.cpu().clone(),)
    return X

def update_Z(X, U):
    new_Z = ()
    idx = 0
    for x, u in zip(X, U):
        z = x + u
        z = projection(z)
        new_Z += (z,)
        idx += 1
    return new_Z

def update_U(U, X, Z):
    new_U = ()
    for u, x, z in zip(U, X, Z):
        new_u = u + x - z
        new_U += (new_u,)
    return new_U

def print_convergence(model, X, Z):
    idx = 0
    print("normalized norm of (weight - projection)")
    for m in model.modules():
        if isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
            x, z = X[idx], Z[idx]
            print("layer {:d}: {:.4f}".format(idx, (x-z).norm().item() / x.norm().item()))
            idx += 1


# CSD quantization with quant_bits = 6
# _QLEVELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 20, 24, 28, 30, 31,
#      -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -12, -14, -15, -16, -17, -18, -20, -24, -28, -30, -31]
_QLEVELS = getQlevels(6)
# print(_QLEVELS)


def train(model, trainloader, testloader, args):
    log = tools.StatLogger(args.log_path)

    # ===================================
    # initialize and run training session
    # ===================================
    model.cuda()
    lossfunc = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=args.weight_decay)

    # retrive pruning masks
    weight_masks = retrive_masks(model)

    # start ADMM training
    Z, U = initialize_Z_and_U(model)
    for epoch in range(args.admm_epochs):  # loop over the dataset multiple times
        epoch += 1
        model.train()
        error_top1 = []
        error_top5 = []
        running_loss = []
        for idx, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = admm_loss(model, lossfunc, Z, U, outputs, labels)
            loss.backward()

            optimizer.step()

            # get masked weights
            apply_mask(model, weight_masks)
            
            error_top1.append(tools.topK_error(outputs, labels, K=1).item())
            error_top5.append(tools.topK_error(outputs, labels, K=5).item())
            running_loss.append(loss.item())

        X = update_X(model)
        Z = update_Z(X, U)
        U = update_U(U, X, Z)
        # print_convergence(model, X, Z)

        error_top1 = np.average(error_top1)
        error_top5 = np.average(error_top5)
        running_loss = np.average(running_loss)
        # print statistics
        print("TRAIN epoch:%-4d error_top1: %.4f error_top5: %.4f loss:%.4f" % (epoch, error_top1, error_top5, running_loss))
        log.report(epoch=epoch,
                   split='TRAIN',
                   error_top5=float(error_top5),
                   error_top1=float(error_top1),
                   loss=float(running_loss))

        validate(model, testloader, lossfunc, log, epoch)

        print('-- saving model check point')
        torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }, os.path.join(args.checkpoint_dir, 'checkpoint_{}.tar'.format(epoch)))

    # Apply quantization
    index = 0
    for m in model.modules():
        if isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
            m.weight.data = quantize(m.weight.data, max(abs(Z[index].numpy().flatten())))
            index += 1

    # Retraining steps
    for epoch in range(args.retraining_epochs):
        epoch += 1
        model.train()
        error_top1 = []
        error_top5 = []
        running_loss = []

        for idx, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = lossfunc(outputs, labels)
            loss.backward()

            optimizer.step()

            # get masked weights
            apply_mask(model, weight_masks)
            
            error_top1.append(tools.topK_error(outputs, labels, K=1).item())
            error_top5.append(tools.topK_error(outputs, labels, K=5).item())
            running_loss.append(loss.item())

        error_top1 = np.average(error_top1)
        error_top5 = np.average(error_top5)
        running_loss = np.average(running_loss)
        # print statistics
        print("RETRAIN epoch:%-4d error_top1: %.4f error_top5: %.4f loss:%.4f" % (epoch, error_top1, error_top5, running_loss))
        log.report(epoch=epoch,
                   split='RETRAIN',
                   error_top5=float(error_top5),
                   error_top1=float(error_top1),
                   loss=float(running_loss))

        # Quantize again
        index = 0
        for m in model.modules():
            if isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
                m.weight.data = quantize(m.weight.data, max(abs(Z[index].numpy().flatten())))
                index += 1

        validate(model, testloader, lossfunc, log, epoch+args.admm_epochs)

        print('-- saving model check point')
        torch.save({
                'epoch': epoch+args.admm_epochs,
                'state_dict': model.state_dict(),
            }, os.path.join(args.checkpoint_dir, 'checkpoint_{}.tar'.format(epoch+args.admm_epochs)))

    print('Finished Retraining')

def validate(model, testloader, lossfunc, log, epoch):
    print("-- running evaluation on validation split")
    model.eval()
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
        print("VALID epoch:%-4d error_top1: %.4f error_top5: %.4f loss:%.4f" % (epoch, error_top1, error_top5, vld_loss))
        log.report(epoch=epoch,
                    split='VALID',
                    error_top5=float(error_top5),
                    error_top1=float(error_top1),
                    loss=float(vld_loss))


def main():
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        trainloader, testloader = datasets.get_cifar10(args.batch_size)
    if args.dataset == 'cifar100':
        trainloader, testloader = datasets.get_cifar100(args.batch_size)

    if args.arch == 'resnet20':
        model = nets.RESNET20MULT3()
    if args.arch == 'vggnaga':
        model = nets.VGGnagaMULT()
    if args.arch == 'cnnc':
        model = nets.CNNCMULT3()

    # load pretrained checkpoint
    if args.pretrained_weights is not None:
        print("Loading checkpoint '{}'".format(args.pretrained_weights))
        pretrained_ckpt = torch.load(args.pretrained_weights)
        model.load_state_dict(pretrained_ckpt['state_dict'])
        print("Loaded checkpoint '{}'".format(args.pretrained_weights))

    # setup checkpoint directory
    if not pth.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    train(model, trainloader, testloader, args)

if __name__ == '__main__':
    main()
