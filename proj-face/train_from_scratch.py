import argparse
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Subset

from train_from_scratch_utils import AverageMeter, ProgressMeter, accuracy, Summary

import torch.multiprocessing # deal with "Too many open files error" bug
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
# arguments
class Args():
    def __init__(self):
        return
args = Args()

args.distributed = True
args.pretrained = True
SAVE_EVERY_EPOCH = True

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
# tuning
for _seed in [77, 777]:
    # for num_ids in [2,4,6,8]:
    # for _num_epoch in [12]:
    for _data_root in ['../../data/face_data/vbsl_50k_obj/', '../../data/face_data/vbsl_50k/']:
    # for _data_root in ['../../data/face_data/vbsl_50k/']:
        args.arch = 'resnet50'
        args.batch_size = 256
        args.print_freq = 30
        args.momentum = 0.9
        args.weight_decay = 1e-4
        args.num_workers = 4
        
        ### things to change ###
        SAVE_STEP_SIZE = 15
        CUR_STEP = 1
        
        ## control epochs for num_ids
        args.num_ids = None
        if args.num_ids is not None:
            num_outputs = args.num_ids
        else:
            num_outputs = 2
        
        ## change data root
        args.data_root = _data_root 
        
        ## change seed
        args.seed = _seed
        
        ## subset training data
        args.subset_ratio = 1
        
        ## change lr & step_size for ft
        args.lr = 0.001
        
        # args.step_size = int(_num_epoch/3)
        args.step_size = 1000
        
        # args.epochs = _num_epoch
        # args.epochs = int(120 / num_ids)
        args.epochs = 2
        
        ## change name
        args.filename_prefix = f'{args.arch}_finetune_{_data_root.split("/")[-2]}_seed{_seed}_'
        print(args.filename_prefix)
        ### things to change ###

        ### init
        # random seed
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # init model
        if args.arch == 'resnet50':
            model = models.__dict__['resnet50'](pretrained=args.pretrained)
            model.fc = nn.Linear(2048, num_outputs)
        elif args.arch == 'simplecnn':
            from models.simplecnn import SimpleCNN
            model = SimpleCNN()
        elif args.arch == 'resnet50barlowtwins':
            model = models.__dict__['resnet50'](pretrained=False)
            ckpt = torch.load('./zoo/resnet50-barlowtwins.pth')
            model.load_state_dict(ckpt, strict=False)
            model.fc = nn.Linear(2048, num_outputs)
            print(f"loaded barlowtwins resnet50")
        elif args.arch == 'rn50_preIN_notexture_sizeVar':
            # init
            model = models.__dict__['resnet50'](pretrained=False)
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(2048, 1000)
            model = nn.DataParallel(model).cuda()
            # load
            ckpt = torch.load('../mycode/saved_models/rn50_preIN_notexture_sizeVar_model_best.pth.tar')
            model.load_state_dict(ckpt['state_dict'])
            # roll back
            model = model.module
            model.fc = nn.Linear(2048, num_outputs)
            print(f"loaded rn50_preIN_notexture_sizeVar")
        else:
            assert False, f"unsupported model arch: {args.arch}"

        if args.distributed:
            model = nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        # init optim/loss
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # scheduler
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.1)


        # Data loading code
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        _trans_train = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ])
        _trans_test = transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])
        print("init dataset...")
        val_dataset = datasets.ImageFolder(valdir, _trans_test)
        
        
        train_dataset = datasets.ImageFolder(traindir, _trans_train)
        
        ### manipulating train set ###
        
        if args.subset_ratio != 1.0: # support randomly subset the dataset
            from torch.utils.data import SubsetRandomSampler
            # Get the number of images in the dataset
            num_images = len(train_dataset)
            # Create indices for the images in the dataset
            indices = list(range(num_images))
            # Randomly shuffle the indices
            indices_shuffled = torch.randperm(num_images)
            # Split the indices into training and validation sets
            split = int(num_images * args.subset_ratio)
            train_indices = indices_shuffled[:split]
            # Create samplers for the training and validation sets
            train_sampler = SubsetRandomSampler(train_indices)
            print(f"subset ratio = {args.subset_ratio}, sampled from {num_images} to {split}")
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, 
                num_workers=args.num_workers, sampler=train_sampler)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers)
            print("num images:", len(train_loader)*args.batch_size)
            assert len(train_dataset.class_to_idx)==2 and len(val_dataset.class_to_idx)==2, f"{len(train_dataset.class_to_idx)}"
        elif args.num_ids is not None: # support selecting num of ids # do not support both subset and num_ids now
            from torch.utils.data import Subset
            # select the indices of all other folders
            assert len(train_dataset.class_to_idx) == 8
            label_set = set(range(args.num_ids))
            # train set
            idx = [i for i in range(len(train_dataset)) if train_dataset.imgs[i][1] in label_set]
            new_train_dataset = Subset(train_dataset, idx)
            print(f"original train size: {len(train_dataset)}, subset size: {len(new_train_dataset)}")
            train_loader = torch.utils.data.DataLoader(
                new_train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers)
            # val set
            idx = [i for i in range(len(val_dataset)) if val_dataset.imgs[i][1] in label_set]
            new_val_dataset = Subset(val_dataset, idx)
            val_loader = torch.utils.data.DataLoader(
                new_val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers)
            print(f"original val size: {len(val_dataset)}, subset size: {len(new_val_dataset)}")
        else: # standard
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers)
            assert len(train_dataset.class_to_idx)==2 and len(val_dataset.class_to_idx)==2, f"{len(train_dataset.class_to_idx)}"

        def train(train_loader, model, criterion, optimizer, epoch, args):
            global CUR_STEP
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            progress = ProgressMeter(
                len(train_loader),
                [batch_time, data_time, losses, top1],
                prefix="Epoch: [{}]".format(epoch))

            # switch to train mode
            model.train()

            end = time.time()
            for i, (images, target) in enumerate(train_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                images = images.float().cuda()
                target = target.cuda()
                
                # compute output
                output = model(images)
                
                # get loss
                loss = criterion(output, target)
                # loss = torch.autograd.Variable(loss, requires_grad = True)

                # measure accuracy and record loss
                acc_list = accuracy(output, target, topk=(1,))
                acc1 = acc_list[0]
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)
                    
                if SAVE_STEP_SIZE != 0: # support saving by num of steps
                    if CUR_STEP % SAVE_STEP_SIZE == 0: # save model
                        _state = {
                            'epoch': epoch + 1,
                            'arch': 'resnet50',
                            'state_dict': model.state_dict(),
                            'best_acc1': best_acc1,
                            'optimizer' : optimizer.state_dict(),
                            'scheduler' : scheduler.state_dict()
                            }
                        filename=args.filename_prefix+f'step{CUR_STEP}_checkpoint.pth.tar'
                        torch.save(_state, filename)
                        print(f"saving to {filename}")
                    CUR_STEP += 1


        def validate(val_loader, model, criterion, args):

            def run_validate(loader, base_progress=0):
                with torch.no_grad():
                    end = time.time()
                    for i, (images, target) in enumerate(loader):
                        i = base_progress + i

                        images = images.float().cuda()
                        target = target.cuda()

                        # compute output
                        output = model(images)
                        loss = criterion(output, target)

                        # measure accuracy and record loss
                        acc_list = accuracy(output, target, topk=(1,))
                        acc1 = acc_list[0]
                        losses.update(loss.item(), images.size(0))
                        top1.update(acc1[0], images.size(0))

                        # measure elapsed time
                        batch_time.update(time.time() - end)
                        end = time.time()

                        if i % args.print_freq == 0:
                            progress.display(i + 1)

            batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
            losses = AverageMeter('Loss', ':.4e', Summary.NONE)
            top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
            progress = ProgressMeter(
                len(val_loader),
                [batch_time, losses, top1],
                prefix='Test: ')

            # switch to evaluate mode
            model.eval()

            run_validate(val_loader)

            progress.display_summary()

            return top1.avg


        best_acc1 = 0

        for epoch in range(args.epochs):
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args)

            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)
            if args.step_size!=0:
                scheduler.step()

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            _state = {
                'epoch': epoch + 1,
                'arch': 'resnet50',
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
                }
            filename=args.filename_prefix+f'epoch{epoch}_checkpoint.pth.tar'
            if SAVE_EVERY_EPOCH:
                torch.save(_state, filename)
            if is_best:
                torch.save(_state, filename)
                shutil.copyfile(filename, args.filename_prefix+'model_best.pth.tar')
            if SAVE_STEP_SIZE != 0:
                if epoch >= 1:
                    SAVE_STEP_SIZE = 0