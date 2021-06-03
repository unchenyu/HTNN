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
# Training configuration params
# =====================================
parser = argparse.ArgumentParser(description='PyTorch HTNN Training')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size (default: 256)')
parser.add_argument('--admm_epochs', type=int, default=30,
                    help='Number of ADMM pruning epochs (default: 30')
parser.add_argument('--retraining_epochs', type=int, default=150,
                    help='Number of retraining epochs (default: 150')
parser.add_argument('--base_lr', type=float, default=0.1,
                    help='Base learning rate (default: 0.1')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay rate (default: 0.0005')
parser.add_argument('--pretrained_weights', type=str, default='../example/checkpoints/checkpoint_200.tar',
                    help='Path to pretrained model')
parser.add_argument('--checkpoint_dir', type=str, default='../example/ckpt_prune',
                    help='Folder to save checkpoints')
parser.add_argument('--log_path', type=str, default='../example/prune.log',
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
    if epoch > 150:
        lr *= 1e-3
    elif epoch > 100:
        lr *= 1e-2
    elif epoch > 50:
        lr *= 1e-1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def apply_mask(model, weight_masks):
    idx = 0
    for m in model.modules():
        if isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
            m.weight.data *= weight_masks[idx]
            idx += 1

def split_2_groups(filter_groups, weight):
    n_filter = list(weight.size())[0]
    # get cross correlation
    groups = n_filter//2
    cross_corr = np.zeros((groups, groups))
    corr = np.zeros(groups)
    for i in range(0, groups):
        corr[i] = (abs(weight[i,:,:,:]*weight[i,:,:,:])).sum().item()
        for j in range(0, groups):
            cross_corr[i,j] = (abs(weight[i,:,:,:]*weight[j+groups,:,:,:])).sum().item()

    filter_group = []
    index1_sort = np.argsort(corr)[::-1]
    # pair filters with minimum correlation
    for i in range(0, groups):
        idx1 = index1_sort[i]
        corr_tmp = cross_corr[idx1]
        index2 = np.where(corr_tmp==np.min(corr_tmp[np.nonzero(corr_tmp)]))
        index2 = index2[0].item()
        idx2 = index2 + groups
        cross_corr[:,index2] = 0
        filter_group.append((idx1, idx2))
    filter_groups.append(filter_group)
    return filter_groups

def split_3_groups(filter_groups, weight):
    n_filter = list(weight.size())[0]
    groups = n_filter//3
    # get cross correlation
    cross_corr12 = np.zeros((groups, groups))
    cross_corr13 = np.zeros((groups, groups))
    corr = np.zeros(groups)
    for i in range(0, groups):
        corr[i] = (abs(weight[i,:,:,:]*weight[i,:,:,:])).sum().item()
        for j in range(0, groups):
            cross_corr12[i,j] = (abs(weight[i,:,:,:]*weight[j+groups,:,:,:])).sum().item()
            cross_corr13[i,j] = (abs(weight[i,:,:,:]*weight[j+2*groups,:,:,:])).sum().item()

    filter_group = []
    index1_sort = np.argsort(corr)[::-1]
    for i in range(0, groups):
        idx1 = index1_sort[i]
        corr_tmp12 = cross_corr12[idx1]
        corr_tmp13 = cross_corr13[idx1]
        index2 = np.where(corr_tmp12==np.min(corr_tmp12[np.nonzero(corr_tmp12)]))
        index3 = np.where(corr_tmp13==np.min(corr_tmp13[np.nonzero(corr_tmp13)]))
        index2 = index2[0].item()
        index3 = index3[0].item()
        idx2 = index2+groups
        idx3 = index3+2*groups
        cross_corr12[:,index2] = 0
        cross_corr13[:,index3] = 0
        filter_group.append((idx1, idx2, idx3))
    filter_groups.append(filter_group)
    return filter_groups

# structre pruning strategy for 2-transform HTNN, 2 filters share one kernel
def htnn2(model):
    index = 0
    filter_groups = []
    for m in model.modules():
        if isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
            weight = m.weight.data
            filter_groups = split_2_groups(filter_groups, weight)
            index += 1
    return filter_groups

# structre pruning strategy for 3-transform HTNN, 3 filters share one kernel
def htnn3(model):
    index = 0
    filter_groups = []
    for m in model.modules():
        if isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
            weight = m.weight.data
            # only 2 filters share one kernel in the first layer
            if index == 0:
                filter_groups = split_2_groups(filter_groups, weight)
            # 3 filters share one kernel
            else:
                filter_groups = split_3_groups(filter_groups, weight)
            index += 1
    return filter_groups

def projection(weight, filter_group):
    weight_mask = torch.ones(weight.size())
    for group in filter_group:
        if len(group) == 2:
            weight_mask[group[0]] = (abs(weight[group[0]]) > abs(weight[group[1]])).float()
            weight_mask[group[1]] = 1 - weight_mask[group[0]]
        elif len(group) == 3:
            weight_mask[group[0]] = ((abs(weight[group[0]]) > abs(weight[group[1]])) & (abs(weight[group[0]]) > abs(weight[group[2]]))).float()
            weight_mask[group[1]] = ((abs(weight[group[1]]) > abs(weight[group[0]])) & (abs(weight[group[1]]) > abs(weight[group[2]]))).float()
            weight_mask[group[2]] = 1 - weight_mask[group[0]] - weight_mask[group[1]]

    weight *= weight_mask
    return weight

def pruning(model, filter_groups):
    weight_masks = []
    index = 0

    for m in model.modules():
        if isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
            weight = m.weight.data
            num = torch.numel(weight)

            weight_mask = torch.ones(weight.size())
            filter_group = filter_groups[index]
            for group in filter_group:
                if len(group) == 2:
                    weight_mask[group[0]] = (abs(weight[group[0]]) > abs(weight[group[1]])).float()
                    weight_mask[group[1]] = 1 - weight_mask[group[0]]
                elif len(group) == 3:
                    weight_mask[group[0]] = ((abs(weight[group[0]]) > abs(weight[group[1]])) & (abs(weight[group[0]]) > abs(weight[group[2]]))).float()
                    weight_mask[group[1]] = ((abs(weight[group[1]]) > abs(weight[group[0]])) & (abs(weight[group[1]]) > abs(weight[group[2]]))).float()
                    weight_mask[group[2]] = 1 - weight_mask[group[0]] - weight_mask[group[1]]

            weight_mask = weight_mask.cuda()
            weight_masks.append(weight_mask)
            index += 1

            m.weight.data *= weight_mask

    return weight_masks

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
        if isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
            Z += (m.weight.data.cpu().clone(),)
            U += (torch.zeros_like(m.weight.data).cpu(),)
    return Z, U

def update_X(model):
    X = ()
    for m in model.modules():
        if isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
            X += (m.weight.data.cpu().clone(),)
    return X

def update_Z(X, U, filter_groups):
    new_Z = ()
    idx = 0
    for x, u in zip(X, U):
        z = x + u
        z = projection(z, filter_groups[idx])
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


def train(model, trainloader, testloader, args):
    log = tools.StatLogger(args.log_path)

    # ===================================
    # initialize and run training session
    # ===================================
    model.cuda()
    lossfunc = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=args.weight_decay)

    # get filter groups
    if args.arch == 'vggnaga':
        filter_groups = htnn2(model)
    else:
        filter_groups = htnn3(model)

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
            
            error_top1.append(tools.topK_error(outputs, labels, K=1).item())
            error_top5.append(tools.topK_error(outputs, labels, K=5).item())
            running_loss.append(loss.item())

        X = update_X(model)
        Z = update_Z(X, U, filter_groups)
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
    
    # Obtain pruning masks
    weight_masks = pruning(model, filter_groups)

    ele_count = 0
    one_count = 0
    for mask in weight_masks:
        ele_count = ele_count + mask.cpu().numpy().size
        one_count = one_count + np.sum(mask.cpu().numpy())
    print('compress rate: %.4f total weights: %d left weights: % d ' % (1 - one_count*1.0 / ele_count, ele_count, one_count))
    log.report(epoch=epoch,
               compress_rate=float(1 - one_count*1.0 / ele_count),
               total_weights=float(ele_count),
               left_weights=float(one_count))

    # Retraining steps
    for epoch in range(args.retraining_epochs):
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
