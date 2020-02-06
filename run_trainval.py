#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
This script is used for training and inferencing our deep neural network.
"""

import os
import shutil
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import datasets
import models3D

from util.os_utils import mkdir_p, isfile, isdir
from util.train_utils import save_checkpoint, adjust_learning_rate, AverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    global device
    lowest_loss = 1e40

    # create checkpoint dir and log dir
    if not isdir(args.checkpoint):
        print("Create new checkpoint folder " + args.checkpoint)
    mkdir_p(args.checkpoint)
    if not args.resume:
        if isdir(args.logdir):
            shutil.rmtree(args.logdir)
        mkdir_p(args.logdir)
    if not args.evaluate:
        logger = SummaryWriter(args.logdir)

    # create model
    print("==> creating model")
    input_channel = 1
    if 'curvature' in args.input_feature:
        input_channel += 2
    if 'vertex_kde' in args.input_feature:
        input_channel += 1
    if 'sd' in args.input_feature:
        input_channel += 1
    n_stack = args.num_stack

    model = models3D.__dict__[args.arch](input_channels=input_channel, n_stack=n_stack)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs')
        model = nn.DataParallel(model)
    model.to(device)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    train_loader = torch.utils.data.DataLoader(datasets.Heatmap3D_sdf(args.data_path, args.json_file, 'train', args.kde, args.input_feature),
                                               batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(datasets.Heatmap3D_sdf(args.data_path, args.json_file, 'val', args.kde, args.input_feature),
                                             batch_size=args.test_batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(datasets.Heatmap3D_sdf(args.data_path, args.json_file, 'test', args.kde, args.input_feature),
                                              batch_size=args.test_batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_loss_joint, test_loss_bone = test(test_loader, model, args)
        print('test loss: ', test_loss, 'test_loss_joint: ', test_loss_joint, 'test_loss_bone: ', test_loss_bone)
        #args.output_dir = args.output_dir + '_val'
        #val_loss, val_loss_joint, val_loss_bone = test(val_loader, model, args)
        #print('val loss: ', val_loss, 'val_loss_joint: ', val_loss_joint, 'val_loss_bone: ', val_loss_bone)
        return

    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
        train_loss, train_loss_joint, train_loss_bone = train(train_loader, model, optimizer)
        valid_loss, val_loss_joint, val_loss_bone = validate(val_loader, model)
        print(valid_loss, val_loss_joint, val_loss_bone)
        test_loss, test_loss_joint, test_loss_bone = test(test_loader, model, args)
        print(test_loss, test_loss_joint, test_loss_bone)

        # remember best acc and save checkpoint
        is_best = valid_loss < lowest_loss
        lowest_loss = min(valid_loss, lowest_loss)
        if torch.cuda.device_count() > 1:
            states = {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'lowest_loss': lowest_loss,
                'optimizer': optimizer.state_dict(),
            }
        else:
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'lowest_loss': lowest_loss,
                'optimizer': optimizer.state_dict(),
            }

        save_checkpoint(states, is_best, checkpoint=args.checkpoint)

        info = {'train_loss': train_loss, 'train_loss_joint': train_loss_joint, 'train_loss_bone': train_loss_bone,
                'val_loss': valid_loss, 'val_loss_joint': val_loss_joint, 'val_loss_bone': val_loss_bone,
                'test_loss': test_loss, 'test_loss_joint': test_loss_joint, 'test_loss_bone': test_loss_bone}
        for tag, value in info.items():
            logger.add_scalar(tag, value, epoch+1)


def train(train_loader, model, optimizer):
    global device
    model.train()  # switch to train mode
    losses = AverageMeter()
    losses_joint = AverageMeter()
    losses_bone = AverageMeter()
    for i, (inputs, mask, target_joint, target_bone, meta) in enumerate(train_loader):
        inputs_var = inputs.to(device)
        target_joint_var = target_joint[:, None, :, :, :].to(device)
        target_bone_var = target_bone[:, None, :, :, :].to(device)
        mask_var = mask[:, None, :, :, :].to(device)

        (score_map_joint, score_map_bone) = model(inputs_var, meta['min_5_fs'].view(-1, 1, 1, 1, 1).float().to(device))
        loss_joint, loss_bone = 0.0, 0.0
        for n_stack in range(len(score_map_joint)):
            loss_joint += F.binary_cross_entropy_with_logits(score_map_joint[n_stack], target_joint_var,
                                                             weight=mask_var, reduction='sum') / mask_var.sum()
            loss_bone += F.binary_cross_entropy_with_logits(score_map_bone[n_stack], target_bone_var,
                                                            weight=mask_var, reduction='sum') / mask_var.sum()
        loss = loss_joint + loss_bone

        #record loss
        losses_joint.update(loss_joint.item())
        losses_bone.update(loss_bone.item())
        losses.update(loss.item())
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    return losses.avg, losses_joint.avg, losses_bone.avg


def validate(val_loader, model):
    global device
    losses = AverageMeter()
    losses_joint = AverageMeter()
    losses_bone = AverageMeter()
    model.eval()  # switch to test mode
    with torch.no_grad():
        for i, (inputs, mask, target_joint, target_bone, meta) in enumerate(val_loader):
            inputs_var = inputs.to(device)
            target_joint_var = target_joint[:, None, :, :, :].to(device)
            target_bone_var = target_bone[:, None, :, :, :].to(device)
            mask_var = mask[:, None, :, :, :].to(device)

            (score_map_joint, score_map_bone) = model(inputs_var, meta['min_5_fs'].view(-1, 1, 1, 1, 1).float().to(device))

            loss_joint, loss_bone = 0.0, 0.0
            for n_stack in range(len(score_map_joint)):
                loss_joint += F.binary_cross_entropy_with_logits(score_map_joint[n_stack], target_joint_var,
                                                                 weight=mask_var, reduction='sum') / mask_var.sum()
                loss_bone += F.binary_cross_entropy_with_logits(score_map_bone[n_stack], target_bone_var,
                                                                weight=mask_var, reduction='sum') / mask_var.sum()

            loss = loss_joint + loss_bone
            # record loss
            losses_joint.update(loss_joint.item())
            losses_bone.update(loss_bone.item())
            losses.update(loss.item())

    return losses.avg, losses_joint.avg, losses_bone.avg


def test(test_loader, model, args):
    global device
    losses = AverageMeter()
    losses_joint = AverageMeter()
    losses_bone = AverageMeter()
    model.eval()  # switch to test mode
    with torch.no_grad():
        for i, (inputs, mask, target_joint, target_bone, meta) in enumerate(test_loader):
            inputs_var = inputs.to(device)
            target_joint_var = target_joint[:, None, :, :, :].to(device)
            target_bone_var = target_bone[:, None, :, :, :].to(device)
            mask_var = mask[:, None, :, :, :].to(device)

            (score_map_joint_raw, score_map_bone_raw) = model(inputs_var, meta['min_5_fs'].view(-1, 1, 1, 1, 1).float().to(device))

            if args.evaluate:
                output_folder = os.path.join('results/', args.output_dir)
                if not os.path.isdir(output_folder):
                    mkdir_p(output_folder)
                score_map_joint = torch.sigmoid(score_map_joint_raw[-1])
                score_map_bone = torch.sigmoid(score_map_bone_raw[-1])
                score_map_joint *= mask_var
                score_map_bone *= mask_var
                score_map_joint = score_map_joint.cpu().data.numpy().squeeze()
                score_map_bone = score_map_bone.cpu().data.numpy().squeeze()

                inputs_bin = (inputs[:, 0, ...] < 0)
                inputs_bin = inputs_bin.squeeze()
                for id in range(len(inputs)):
                    name_id = meta['name'][id]
                    np.save(os.path.join(output_folder, 'input_' + name_id + '.npy'), inputs_bin[id])
                    np.save(os.path.join(output_folder, 'joint_pred_' + name_id + '.npy'), score_map_joint[id])
                    np.save(os.path.join(output_folder, 'bone_pred_'+ name_id +'.npy'), score_map_bone[id])
                    with open(os.path.join(output_folder,'ts_' + name_id +'.txt'), 'w') as f_ts:
                        f_ts.write('{0} {1} {2}\n'.format(meta['center_trans'][0][id].item(),
                                                          meta['center_trans'][1][id].item(),
                                                          meta['center_trans'][2][id].item()))
                        f_ts.write('{0} {1} {2}\n'.format(meta['translate'][0][id].item(),
                                                          meta['translate'][1][id].item(),
                                                          meta['translate'][2][id].item()))
                        f_ts.write('{0}\n'.format(meta['scale'][id].item()))

            loss_joint = F.binary_cross_entropy_with_logits(score_map_joint_raw[-1], target_joint_var,
                                                            weight=mask_var, reduction='sum') / mask_var.sum()
            loss_bone = F.binary_cross_entropy_with_logits(score_map_bone_raw[-1], target_bone_var,
                                                           weight=mask_var, reduction='sum') / mask_var.sum()
            loss = loss_joint + loss_bone

            # # record loss
            losses_joint.update(loss_joint.item())
            losses_bone.update(loss_bone.item())
            losses.update(loss.item())

    return losses.avg, losses_joint.avg, losses_bone.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch 3D Heatmap Training')
    # Training
    parser.add_argument('--arch', '-a', metavar='ARCH', default='v2v_hg', help='model architecture.')
    parser.add_argument('--num_stack', metavar='NT', type=int, default=4, help='number of hourglass module.')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='epoch number')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('-j', '--workers', default=3, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[], help='Decrease learning rate at these epochs.')
    parser.add_argument('--kde', type=float, default=10,
                        help='sigma of gaussian around each surface vertex is kde times average edge length')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model')
    parser.add_argument('--train-batch', default=2, type=int, metavar='N', help='train batchsize')
    parser.add_argument('--test-batch', default=2, type=int, metavar='N', help='test batchsize')
    parser.add_argument('-c', '--checkpoint', default='checkpoints/volNet', type=str, metavar='PATH',
                        help='path to save checkpoint')
    parser.add_argument('--logdir', default='logs/volNet', type=str, metavar='LOG', help='directory to save logs')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--data_path', type=str, help='h5 data file with all data',
                        default='model_resource_data/model-resource-volumetric.h5')
    parser.add_argument('--json_file', type=str, help='annotation json file',
                        default='model_resource_data/model-resource-volumetric.json')
    parser.add_argument('--output_dir', type=str, default='volNet', help='prediction output folder')
    parser.add_argument('--input_feature', type=str, nargs='+', default=['curvature', 'sd', 'vertex_kde'],
                        help='input feature name list (curvature, sd, vertex_kde)')

    print(parser.parse_args())
    main(parser.parse_args())

