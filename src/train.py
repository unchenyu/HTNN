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

# =====================================
# Training configuration params
# =====================================
parser = argparse.ArgumentParser(description='PyTorch HTNN Training')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size (default: 256)')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='Number of training epochs (default: 200')
parser.add_argument('--base_lr', type=float, default=0.1,
                    help='Base learning rate (default: 0.1')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay rate (default: 0.0005')
parser.add_argument('--pretrained_weights', type=str, default=None,
                    help='Path to pretrained model')
parser.add_argument('--checkpoint_dir', type=str, default='../example/checkpoints',
                    help='Folder to save checkpoints')
parser.add_argument('--log_path', type=str, default='../example/train.log',
                    help='Path to write logs into')
model_names = ['resnet20', 'vggnaga', 'cnnc']
dataset_names = ['cifar10', 'cifar100']
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture (default: resnet20)')
parser.add_argument('--dataset', metavar='DATA', default='cifar10',
                    choices=dataset_names,
                    help='dataset (default: cifar10')


def lr_schedule_vgg(optimizer, epoch, base_lr=0.1):
    lr = base_lr
    if epoch >= 50:
        lr = base_lr * (0.5 ** (epoch // 25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def lr_schedule_resnet(optimizer, epoch, base_lr=0.1):
    lr = base_lr
    if epoch > 200:
        lr *= 1e-3
    elif epoch > 150:
        lr *= 1e-2
    elif epoch > 100:
        lr *= 1e-1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model, trainloader, testloader, args):
    log = tools.StatLogger(args.log_path)

    # ===================================
    # initialize and run training session
    # ===================================
    model.cuda()

    lossfunc = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=args.weight_decay)
    
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times
        if args.arch == 'resnet20':
            lr_schedule_resnet(optimizer, epoch, args.base_lr)
        else:
            lr_schedule_vgg(optimizer, epoch, args.base_lr)
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

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
            
            error_top1.append(tools.topK_error(outputs, labels, K=1).item())
            error_top5.append(tools.topK_error(outputs, labels, K=5).item())
            running_loss.append(loss.item())

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

        print('-- saving model check point')
        torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }, os.path.join(args.checkpoint_dir, 'checkpoint_{}.tar'.format(epoch)))
                
    print('Finished Training')


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
