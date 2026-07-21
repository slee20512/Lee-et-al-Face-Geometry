import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from train_from_scratch_utils import AverageMeter, ProgressMeter, accuracy, Summary

import torch.multiprocessing # deal with "Too many open files error" bug
torch.multiprocessing.set_sharing_strategy('file_system')

cudnn.benchmark = True

# Curriculum learning, stage 2 (supervised branch): take the supervised, from-scratch
# 60-way-IDEM backbone (train_from_scratch_60way.py's SL_resnet50_scratch_60way_IDEM_colorbg_seed777_model_best)
# and continue supervised training at a much lower lr on one identity's 7-way emotion task.
# Swaps in a fresh 7-class head since the stage-1 checkpoint is 60-way.

# arguments
class Args():
    def __init__(self):
        return
args = Args()

SAVE_EVERY_N_EPOCHS = 5  # periodic snapshot cadence; 'latest' and 'best' are always saved too

args.arch = 'resnet50'
args.batch_size = 128
args.print_freq = 30
args.momentum = 0.9
args.weight_decay = 1e-4
args.num_workers = 8

args.stage1_checkpoint = 'resnet50_scratch_60way_IDEM_colorbg_seed777_model_best.pth.tar'
args.stage1_num_classes = 60

args.data_root = '/mnt/smb/locker/issa-locker/users/Seojin/data/face_data/vbsle_50k_texture_colorbg/7way_EM_dan_colorbg/'
args.seed = 777

# much lower lr than the stage-1 from-scratch schedule (which started at 0.1) --
# the backbone is already specialized for this face-ID/emotion domain, we're just
# adapting the head + lightly nudging features toward one identity's 7 emotions.
args.lr = 1e-4
args.epochs = 20
args.lr_milestones = [10, 16]

args.filename_prefix = f'{args.arch}_curriculum_sl_7way_EM_dan_colorbg_from_scratch60way_seed{args.seed}_'
print(args.filename_prefix)

random.seed(args.seed)
torch.manual_seed(args.seed)

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
train_dataset = datasets.ImageFolder(traindir, _trans_train)
val_dataset = datasets.ImageFolder(valdir, _trans_test)
num_outputs = len(train_dataset.classes)
print(f"{num_outputs} classes: {train_dataset.classes}")
assert num_outputs == 7, f"expected 7 emotion classes, got {num_outputs}: {train_dataset.classes}"

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True)

criterion = nn.CrossEntropyLoss().cuda()
scaler = torch.cuda.amp.GradScaler()

start_epoch = 0
best_acc1 = 0
latest_filename = args.filename_prefix + 'latest_checkpoint.pth.tar'
if os.path.exists(latest_filename):
    print(f"resuming from {latest_filename}")
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    ckpt = torch.load(latest_filename, map_location='cuda')
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch']
    best_acc1 = ckpt['best_acc1']
    print(f"resumed at epoch {start_epoch}, best_acc1 {best_acc1}")
else:
    print(f"initializing from stage-1 checkpoint {args.stage1_checkpoint}")
    # stage-1 checkpoint is 60-way; build with that head to match the saved
    # state_dict, then swap in a fresh 7-way head.
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, args.stage1_num_classes)
    stage1_ckpt = torch.load(args.stage1_checkpoint, map_location='cuda')
    state_dict = stage1_ckpt['state_dict']
    # strip 'module.' in case stage 1 was trained with DataParallel
    state_dict = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.float().cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc_list = accuracy(output, target, topk=(1,))
        acc1 = acc_list[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, args):

    def run_validate(loader):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                images = images.float().cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                with torch.cuda.amp.autocast():
                    output = model(images)
                    loss = criterion(output, target)

                acc_list = accuracy(output, target, topk=(1,))
                acc1 = acc_list[0]
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))

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

    model.eval()
    run_validate(val_loader)
    progress.display_summary()

    return top1.avg


for epoch in range(start_epoch, args.epochs):
    train(train_loader, model, criterion, optimizer, epoch, args)

    acc1 = validate(val_loader, model, criterion, args)
    scheduler.step()

    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    _state = {
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
        }
    torch.save(_state, latest_filename)
    if is_best:
        shutil.copyfile(latest_filename, args.filename_prefix + 'model_best.pth.tar')
    if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
        shutil.copyfile(latest_filename, args.filename_prefix + f'epoch{epoch}_checkpoint.pth.tar')
