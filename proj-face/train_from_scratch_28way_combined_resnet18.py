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
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from train_from_scratch_utils import AverageMeter, ProgressMeter, accuracy, Summary

import torch.multiprocessing # deal with "Too many open files error" bug
torch.multiprocessing.set_sharing_strategy('file_system')

cudnn.benchmark = True

# ResNet-18 ablation of train_from_scratch_28way_combined.py (supervised branch): same
# *combined* 28-way IDEM dataset -- merges the plain vbsle_50k/28way_IDEM set with the
# vbsle_50k_texture_colorbg/28way_IDEM_colorbg set (same 28 identity x emotion classes
# in both, verified matching folder names -- ImageFolder's class_to_idx is computed by
# sorting folder names, so both instances get identical label indices and can be
# concatenated directly) -- but resnet18 instead of resnet50, mirroring
# train_from_scratch_60way_resnet18.py. Unlike the plain-only 28way_IDEM dataset (whose
# resnet18 from-scratch run collapsed to chance -- see
# train_from_scratch_28way_IDEM_resnet18.py and its WSD/warmup+cosine ablations), this
# set mixes in the higher-contrast colorbg images too, so the plain MultiStepLR recipe
# is tried here first rather than assuming it will need the same fix.

# arguments
class Args():
    def __init__(self):
        return
args = Args()

args.pretrained = False  # random init -- no ImageNet weights
SAVE_EVERY_N_EPOCHS = 10  # periodic snapshot cadence; 'latest' and 'best' are always saved too

args.arch = 'resnet18'
args.batch_size = 128
args.print_freq = 30
args.momentum = 0.9
args.weight_decay = 1e-4
args.num_workers = 8

args.data_root_a = '/mnt/smb/locker/issa-locker/users/Seojin/data/face_data/vbsle_50k/28way_IDEM/'
args.data_root_b = '/mnt/smb/locker/issa-locker/users/Seojin/data/face_data/vbsle_50k_texture_colorbg/28way_IDEM_colorbg/'
args.seed = 777

# from-scratch schedule: same as train_from_scratch_28way_combined.py
args.lr = 0.1
args.epochs = 100
args.lr_milestones = [30, 60, 90]

args.filename_prefix = f'{args.arch}_scratch_28way_IDEM_combined_seed{args.seed}_'
print(args.filename_prefix)

random.seed(args.seed)
torch.manual_seed(args.seed)

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
train_dataset_a = datasets.ImageFolder(os.path.join(args.data_root_a, 'train'), _trans_train)
train_dataset_b = datasets.ImageFolder(os.path.join(args.data_root_b, 'train'), _trans_train)
val_dataset_a = datasets.ImageFolder(os.path.join(args.data_root_a, 'val'), _trans_test)
val_dataset_b = datasets.ImageFolder(os.path.join(args.data_root_b, 'val'), _trans_test)

assert train_dataset_a.class_to_idx == train_dataset_b.class_to_idx, \
    f"class_to_idx mismatch between the two datasets -- concatenation would silently corrupt labels: {train_dataset_a.class_to_idx} vs {train_dataset_b.class_to_idx}"
assert val_dataset_a.class_to_idx == train_dataset_a.class_to_idx
assert val_dataset_b.class_to_idx == train_dataset_a.class_to_idx

train_dataset = ConcatDataset([train_dataset_a, train_dataset_b])
val_dataset = ConcatDataset([val_dataset_a, val_dataset_b])
num_outputs = len(train_dataset_a.classes)
print(f"{num_outputs} classes: {train_dataset_a.classes}")
print(f"combined train size: {len(train_dataset)} ({len(train_dataset_a)} + {len(train_dataset_b)})")
print(f"combined val size: {len(val_dataset)} ({len(val_dataset_a)} + {len(val_dataset_b)})")

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True)

# init model -- random init, no ImageNet weights
model = models.__dict__[args.arch](pretrained=args.pretrained)
model.fc = nn.Linear(model.fc.in_features, num_outputs)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.cuda()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
scaler = torch.cuda.amp.GradScaler()

start_epoch = 0
best_acc1 = 0
latest_filename = args.filename_prefix + 'latest_checkpoint.pth.tar'
if os.path.exists(latest_filename):
    print(f"resuming from {latest_filename}")
    ckpt = torch.load(latest_filename, map_location='cuda')
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch']
    best_acc1 = ckpt['best_acc1']
    print(f"resumed at epoch {start_epoch}, best_acc1 {best_acc1}")


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
