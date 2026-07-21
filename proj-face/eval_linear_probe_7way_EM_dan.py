import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.svm import LinearSVC

import torch.multiprocessing # deal with "Too many open files error" bug
torch.multiprocessing.set_sharing_strategy('file_system')

# Linear-probe eval for the dan-curriculum SimCLR checkpoints: all three (lr 3e-5,
# lr 3e-4, from-scratch) converged to the same ~2.91 NT-Xent loss floor. A flat
# contrastive loss doesn't necessarily mean the learned features are useless for
# downstream discrimination -- this checks that directly, the same way the rest of
# this codebase evaluates models (fit a LinearSVC on frozen features, see MyMain_SL.ipynb),
# rather than by staring at the training loss curve. Also includes the pre-curriculum
# 60-way-only SimCLR checkpoint as a baseline, to see whether stage-2 curriculum
# training changed feature separability for this identity's 7 emotions at all.

CHECKPOINTS = {
    'stage1_60way_only (baseline, no curriculum)': 'resnet50_simclr_60way_IDEM_colorbg_seed777_model_best.pth.tar',
    'curriculum_lr3e-5': 'resnet50_curriculum_simclr_7way_EM_dan_colorbg_from_simclr60way_seed777_model_best.pth.tar',
    'curriculum_lr3e-4': 'resnet50_curriculum_simclr_7way_EM_dan_colorbg_from_simclr60way_lr3e-4_seed777_model_best.pth.tar',
    'scratch': 'resnet50_simclr_7way_EM_dan_colorbg_scratch_seed777_model_best.pth.tar',
}

DATA_ROOT = '/mnt/smb/locker/issa-locker/users/Seojin/data/face_data/vbsle_50k_texture_colorbg/7way_EM_dan_colorbg/'
BATCH_SIZE = 256
NUM_WORKERS = 4
HIDDEN_DIM = 1024
PROJ_DIM = 128

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
_trans_eval = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

print("init dataset...")
train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, 'train'), _trans_eval)
val_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, 'val'), _trans_eval)
print(f"train: {len(train_dataset)} images, val: {len(val_dataset)} images, classes: {train_dataset.classes}")

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True)


def build_simclr_model():
    # same construction as the train_*simclr* scripts -- fc is the projector
    model = models.resnet50(pretrained=False)
    feature_dim = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(feature_dim, HIDDEN_DIM),
        nn.ReLU(inplace=True),
        nn.Linear(HIDDEN_DIM, PROJ_DIM),
    )
    return model


@torch.no_grad()
def extract_features(model, loader):
    # probe on the pre-projector representation (avgpool output, 2048-d) -- the
    # SimCLR paper found this transfers better than the projected embedding.
    model.eval()
    feats, labels = [], []
    for images, target in loader:
        images = images.float().cuda(non_blocking=True)
        f = model(images)
        feats.append(f.cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(feats), np.concatenate(labels)


results = {}
for name, checkpoint_path in CHECKPOINTS.items():
    print(f"\n=== {name} ({checkpoint_path}) ===")
    if not os.path.exists(checkpoint_path):
        print(f"missing, skipping: {checkpoint_path}")
        continue

    model = build_simclr_model()
    ckpt = torch.load(checkpoint_path, map_location='cuda')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    state_dict = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.fc = nn.Identity()  # strip the projector, probe on the 2048-d backbone features
    model = model.cuda()

    print("extracting train features...")
    X_train, y_train = extract_features(model, train_loader)
    print("extracting val features...")
    X_val, y_val = extract_features(model, val_loader)

    clf = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=1e-4,
                     fit_intercept=True, C=1.0, max_iter=20000)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    print(f"{name}: train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
    results[name] = (train_acc, val_acc)

print("\n=== summary ===")
for name, (train_acc, val_acc) in results.items():
    print(f"{name:45s}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
