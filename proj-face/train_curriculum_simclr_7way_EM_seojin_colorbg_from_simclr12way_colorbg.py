import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import ImageFilter

from train_from_scratch_utils import AverageMeter, ProgressMeter

import torch.multiprocessing # deal with "Too many open files error" bug
torch.multiprocessing.set_sharing_strategy('file_system')

cudnn.benchmark = True

# Textured-domain mirror of train_curriculum_simclr_7way_EM_seojin_from_simclr12way.py:
# init from the stage-1 textured-12-way-identity SimCLR backbone
# (train_simclr_12way_colorbg_scratch.py), continue NT-Xent training on the textured
# 7-way emotion dataset (vbsle_50k_texture_colorbg/7way_EM_seojin_colorbg) instead of
# the geometry-only 7way_EM_seojin.

# arguments
class Args():
    def __init__(self):
        return
args = Args()

args.arch = 'resnet50'
args.batch_size = 64  # 2*batch_size augmented views go through the network per step -- 128 OOM'd on an 11GB 2080Ti even with AMP
args.print_freq = 30
args.num_workers = 8
args.hidden_dim = 1024
args.proj_dim = 128
args.temperature = 0.5
args.weight_decay = 1e-6
args.seed = 777
SAVE_EVERY_N_EPOCHS = 10  # periodic snapshot cadence; 'latest' and 'best' are always saved too

args.stage1_checkpoint = 'resnet50_simclr_texture_colorbg_12way_seed777_model_best.pth.tar'

args.lr = 3e-5
args.epochs = 50

args.data_root = '/mnt/smb/locker/issa-locker/users/Seojin/data/face_data/vbsle_50k_texture_colorbg/7way_EM_seojin_colorbg/'
args.filename_prefix = f'{args.arch}_curriculum_simclr_7way_EM_seojin_colorbg_from_simclr12way_colorbg_seed{args.seed}_'
print(args.filename_prefix)

random.seed(args.seed)
torch.manual_seed(args.seed)


class GaussianBlur(object):
    """Gaussian blur augmentation, as used in the SimCLR paper (Chen et al., 2020)."""
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(*self.sigma)
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class TwoCropsTransform(object):
    """Returns two independently-augmented views of the same image."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
_simclr_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur((0.1, 2.0))], p=0.5),
    transforms.ToTensor(),
    normalize,
])

print("init dataset...")
traindir = os.path.join(args.data_root, 'train')
train_dataset = datasets.ImageFolder(traindir, TwoCropsTransform(_simclr_aug))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True, drop_last=True)


def build_simclr_model(arch, hidden_dim, proj_dim):
    # keep `model.fc` as the projector (rather than a separate submodule) so checkpoints
    # load directly with the existing `model.fc = projector` convention in load_model.py
    model = models.__dict__[arch](pretrained=False)
    feature_dim = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, proj_dim),
    )
    return model


def nt_xent_loss(z1, z2, temperature):
    """Normalized temperature-scaled cross entropy loss (SimCLR, Chen et al., 2020)."""
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temperature
    sim.fill_diagonal_(torch.finfo(sim.dtype).min)  # -1e9 overflows float16 under autocast
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


scaler = torch.cuda.amp.GradScaler()

start_epoch = 0
best_loss = float('inf')
latest_filename = args.filename_prefix + 'latest_checkpoint.pth.tar'
if os.path.exists(latest_filename):
    print(f"resuming from {latest_filename}")
    model = build_simclr_model(args.arch, args.hidden_dim, args.proj_dim)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    ckpt = torch.load(latest_filename, map_location='cuda')
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch']
    best_loss = ckpt['loss']
    print(f"resumed at epoch {start_epoch}, best_loss {best_loss}")
else:
    print(f"initializing from stage-1 checkpoint {args.stage1_checkpoint}")
    model = build_simclr_model(args.arch, args.hidden_dim, args.proj_dim)
    stage1_ckpt = torch.load(args.stage1_checkpoint, map_location='cuda')
    state_dict = stage1_ckpt['state_dict'] if 'state_dict' in stage1_ckpt else stage1_ckpt
    # strip 'module.' in case stage 1 was trained with DataParallel
    state_dict = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)


def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, ((x1, x2), _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        x1 = x1.float().cuda(non_blocking=True)
        x2 = x2.float().cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            z1 = model(x1)
            z2 = model(x2)
            loss = nt_xent_loss(z1, z2, args.temperature)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), x1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    return losses.avg


for epoch in range(start_epoch, args.epochs):
    epoch_loss = train(train_loader, model, optimizer, epoch, args)
    scheduler.step()

    is_best = epoch_loss < best_loss
    best_loss = min(epoch_loss, best_loss)

    _state = {
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'loss': epoch_loss,
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
        }
    torch.save(_state, latest_filename)
    if is_best:
        shutil.copyfile(latest_filename, args.filename_prefix + 'model_best.pth.tar')
    if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
        shutil.copyfile(latest_filename, args.filename_prefix + f'epoch{epoch}_checkpoint.pth.tar')
