"""
Read PyTorch model from .pth.tar checkpoint.
"""

import os
import sys
<<<<<<< HEAD

=======
import torchfile
>>>>>>> 535f7fc (temp update)
import torch
import torch.nn as nn
from torch.utils import model_zoo

import torchvision
import torchvision.models as models
<<<<<<< HEAD

=======
from collections import OrderedDict
>>>>>>> 535f7fc (temp update)
from rn50_auxiliary_dm import rn50_auxiliary_dm

model_urls = {
            'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
            'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
            'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
            }

def load_model(model_name):
    print(f"loading {model_name}")
    
    # load default IN pretrained models
    if model_name in ['alexnet', 'vgg16', 'resnet50']:
        model = models.__dict__[model_name](pretrained=True).cuda()
    elif model_name == 'vggface':
        model = Vgg_face_dag().cuda()
        ckpt = torch.load('./saved_models/vgg_face_dag.pth')
        model.load_state_dict(ckpt)
<<<<<<< HEAD
    
=======
    elif model_name == 'vggface_finetune_56way_IDEM_colorbg_seed777':
        model = Vgg_face_dag().cuda()
        # Load checkpoint
        ckpt = torch.load('vggface_finetune_56way_IDEM_colorbg_seed777_.pth.tar')

        # Extract the state_dict from the checkpoint
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        # Optional: remove 'module.' prefix if the model was saved from a DataParallel wrapper
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v

        # Load the weights
        model.load_state_dict(new_state_dict, strict=True)

    elif model_name == 'vggface_AppleMesh02':
        # Instantiate Model
        model = InceptionResNet(num_classes=512).cuda()

        # Load Pretrained Weights
        ckpt = torch.load("/mnt/smb/locker/issa-locker/users/Seojin/saved_models/20180402-114759-vggface2-features.pth")
        model.load_state_dict(ckpt, strict=False)

       
    elif model_name == 'vggface_AppleMesh022':
        # Load .t7 model
        t7_path = "/mnt/smb/locker/issa-locker/users/Seojin/saved_models/vgg_face_torch/VGG_FACE.t7"
        checkpoint = torchfile.load(t7_path)

        if checkpoint is None:
            raise ValueError(f"Failed to load model from {t7_path}. The file might be corrupted or incompatible.")

        print("Loaded .t7 checkpoint type:", type(checkpoint))

        # Check if model data is inside `_obj`
        if hasattr(checkpoint, '_obj'):
            checkpoint = checkpoint._obj  # Extract `_obj`

        print("Checkpoint type after extracting _obj:", type(checkpoint))

        # Check if `_obj` is still empty
        if not checkpoint:
            raise ValueError("The checkpoint _obj is empty. The .t7 file may not contain valid model weights.")

        # Extract the parameters manually
        state_dict = OrderedDict()
        if hasattr(checkpoint, 'parameters'):
            params = checkpoint.parameters()
            for i, param in enumerate(params):
                if isinstance(param, torch.Tensor):
                    state_dict[f'layer_{i}'] = param

        # Debugging: Print the extracted keys
        print("Extracted state_dict keys:", list(state_dict.keys()))

        # Ensure the state_dict is not empty
        if not state_dict:
            raise ValueError("Extracted state_dict is empty. Model weights may not be stored in a compatible format.")

        # Load the model
        model = Vgg_face_dag().cuda()  

        # Load the weights into the model
        model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded VGG-Face model from {t7_path}")

    elif model_name == 'facenet':
        # Load FaceNet using InceptionResnetV1 (pretrained on VGGFace2)
        model = InceptionResnetV1(pretrained='vggface2').eval().cuda()
        print("Loaded FaceNet model with VGGFace2 weights.")

    elif model_name == 'facenet_128d':
        # FaceNet 128d → pretrained on CASIA-Webface
        from facenet_pytorch import InceptionResnetV1
        model = InceptionResnetV1(pretrained='casia-webface').eval().cuda()
        print("Loaded FaceNet (128D) pretrained on CASIA-Webface.")
        
    elif model_name == 'facenet_512d':
        # FaceNet 512d → pretrained on VGGFace2
        from facenet_pytorch import InceptionResnetV1
        model = InceptionResnetV1(pretrained='vggface2').eval().cuda()
        print("Loaded FaceNet (512D) pretrained on VGGFace2.")
    
    elif model_name == 'ghostfacenet':
        # GhostFaceNet PyTorch implementation
        from ghostfacenet_pytorch import GhostFaceNet
        model = GhostFaceNet(pretrained=True).eval().cuda()
        print("Loaded GhostFaceNet (PyTorch).")
        
    elif model_name == 'buffalo_l':
        # Buffalo_L (InsightFace PyTorch implementation)
        import insightface
        model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        model.prepare(ctx_id=0)
        print("Loaded Buffalo_L (InsightFace).")

    # https://kilthub.cmu.edu/articles/dataset/Early_experience_with_low-pass_filtered_images_facilitates_visual_category_learning_in_a_neural_network_model/23972115
    elif model_name == 'Tarr_colored':
        model = models.resnet50(pretrained=False)
        state_dict = torch.load('ColoredModel.pt')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"loaded {model_name}")
        model = model.cuda()
    elif model_name == 'Tarr_colored_no_blur':
        model = models.resnet50(pretrained=False)
        state_dict = torch.load('ColoredNoBlurModel.pt')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"loaded {model_name}")
        model = model.cuda()
    elif model_name == 'Tarr_colored_linear_blur':
        model = models.resnet50(pretrained=False)
        state_dict = torch.load('ColoredLinearBlurModel.pt')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"loaded {model_name}")
        model = model.cuda()
    elif model_name == 'Tarr_colored_nonlinear_blur':
        model = models.resnet50(pretrained=False)
        state_dict = torch.load('ColoredNonLinearBlurModel2.pt')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"loaded {model_name}")
        model = model.cuda()
    elif 'lowpassfilter_colored' in model_name:
        model = models.resnet50(pretrained=False)

        # Define the checkpoint path based on the model_name
        if model_name == 'lowpassfilter_colored_noblur':
            checkpoint = torch.load('ColoredImgNet/ColoredImgNet/10/ColoredNoBlurModel.pt')
        elif model_name == 'lowpassfilter_colored_linearblur':
            checkpoint = torch.load('ColoredImgNet/ColoredImgNet/10/ColoredLinearBlurModel.pt')
        elif model_name == 'lowpassfilter_colored_nonlinearblur':
            checkpoint = torch.load('ColoredImgNet/ColoredImgNet/10/ColoredNonLinearBlurModel2.pt')
        elif model_name == 'lowpassfilter_colored':
            checkpoint = torch.load('ColoredImgNet/ColoredImgNet/10/ColoredModel.pt')
        elif model_name == 'lowpassfilter_colored_nonlinearblur1':
            checkpoint = torch.load('ColoredImgNet/ColoredImgNet/1/ColoredNonLinearBlurModel2.pt')
        elif model_name == 'lowpassfilter_colored_nonlinearblur2':
            checkpoint = torch.load('ColoredImgNet/ColoredImgNet/2/ColoredNonLinearBlurModel2.pt')
        # Move model to the correct device (GPU or CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)  # Move the model to the device (important step)

        # Remove 'module.' prefix if present (e.g., from DataParallel)
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

        # Move checkpoint weights to the correct device
        checkpoint = {k: v.to(device) for k, v in checkpoint.items()}

        # Try loading the checkpoint into the model
        try:
            model.load_state_dict(checkpoint, strict=False)  # Use strict=False to allow for mismatched keys
        except RuntimeError as e:
            print(f"Error loading model: {e}")
            print("Some keys may be missing or mismatched.")

        # Adjust the fully connected layer based on the model's task (adjust output size if necessary)
        model.fc = nn.Linear(2048, 86)  # Adjust this line for your task's output size (86 is just an example)

        # Move the fc layer to the same device
        model.fc = model.fc.to(device)

        # Ensure the model is in evaluation mode after loading
        model.eval()

        # Debugging: Print the device of model parameters
        for param_name, param in model.named_parameters():
            print(f"{param_name}: {param.device}")
    elif model_name == 'lowpassfilter_grayscale_linearblur':
        model = models.resnet50(pretrained=False)
        checkpoint = torch.load('BWImgNet/BWImgNet/10/BWLinearBlurModel.pt')

        # Replicate grayscale weights across 3 channels
        with torch.no_grad():
            # Create a new conv1 layer for 3 channels (RGB) and set the weights
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
            # Replicate grayscale weights into the new conv1 layer
            grayscale_weights = checkpoint['module.conv1.weight']  # Shape: [64, 1, 7, 7]
            model.conv1.weight.data = grayscale_weights.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)  # Repeat to 3 channels

        # Remove 'module.' prefix in checkpoint keys (if model was trained with DataParallel)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)  # Move the model to the device (important step)

        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

        # Manually remove the 'conv1.weight' from the checkpoint (since it was already modified)
        checkpoint.pop('conv1.weight', None)

        # Load the rest of the model's state_dict (excluding 'conv1' weights)
        model.load_state_dict(checkpoint, strict=False)

        # Adjust the fully connected layer output size if necessary (for your task)
        model.fc = nn.Linear(2048, 86)

        # Ensure the model is in evaluation mode after loading
        model.eval()

        # Print model parameters device location for verification
        for param_name, param in model.named_parameters():
            print(f"{param_name}: {param.device}")

        # Move the fc layer to the correct device (cuda or cpu)
        model.fc = model.fc.to(device)

        # Ensure the model is in evaluation mode
        model.eval()

        
>>>>>>> 535f7fc (temp update)
    # intermediate layers
    elif 'resnet50_layer' in model_name or 'resnet50_subsampled' in model_name:
        model = models.__dict__['resnet50'](pretrained=True).cuda()
    elif 'alexnet_layer' in model_name:
        model = models.__dict__['alexnet'](pretrained=True).cuda()
    
<<<<<<< HEAD
=======
    elif model_name == 'vbsl50k_colorbg_AppleMesh00_AppleMesh01_contrastive_self_emotion_best_model':
        checkpoint_path = 'vbsl50k_colorbg_AppleMesh00_AppleMesh01_contrastive_self_emotion_best_model.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cuda')

        # 🔹 Load a standard ResNet-50 model
        model = torchvision.models.resnet50()
        print("Loaded standard ResNet-50 architecture.")

        # 🔹 Remove "backbone." prefix from state_dict if necessary
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        backbone_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}

        # 🔹 Load the backbone weights
        model.load_state_dict(backbone_state_dict, strict=False)
        print("Restored backbone weights from Barlow Twins checkpoint.")

        # 🔹 Replace FC layer for classification (if needed)
        model.fc = nn.Linear(in_features=2048, out_features=2)  # Adjust output classes
        print("Replaced final fully connected layer for binary classification.")

        # 🔹 Wrap in DataParallel and move to GPU
        model = nn.DataParallel(model).cuda()
        model.eval()  # Set model to evaluation mode

    elif model_name == 'barlowtwins_class_contrastive_vbsle50k_AppleMesh00_AppleMesh01_contrastive_best_model':
        checkpoint_path = 'barlowtwins_class_contrastive_vbsle50k_AppleMesh00_AppleMesh01_contrastive_best_model.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cuda')

        # 🔹 Load a standard ResNet-50 model
        model = torchvision.models.resnet50()
        print("Loaded standard ResNet-50 architecture.")

        # 🔹 Remove "backbone." prefix from state_dict if necessary
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        backbone_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}

        # 🔹 Load the backbone weights
        model.load_state_dict(backbone_state_dict, strict=False)
        print("Restored backbone weights from Barlow Twins checkpoint.")

        # 🔹 Replace FC layer for classification (if needed)
        model.fc = nn.Linear(in_features=2048, out_features=2)  # Adjust output classes
        print("Replaced final fully connected layer for binary classification.")

        # 🔹 Wrap in DataParallel and move to GPU
        model = nn.DataParallel(model).cuda()
        model.eval()  # Set model to evaluation mode

    elif model_name == 'simclr_vbsle_50k_contrastive_contrastive_best_model':
        checkpoint_path = 'simclr_vbsle_50k_contrastive_contrastive_best_model.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cuda')

        # Load ResNet-50 backbone (not pretrained, since SimCLR trains from scratch)
        model = torchvision.models.resnet50(weights=None)  # No default weights
        print("Loaded ResNet-50 backbone for SimCLR.")

        # Define projection head (must match training)
        projector = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128)  # Projection dimension (must match Args)
        )

        # Combine model and projection head
        model.fc = projector  # Attach the projector in place of FC layer

        # Load the model state dict
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print("Restored weights from SimCLR checkpoint.")

        # Wrap in DataParallel and move to GPU
        model = nn.DataParallel(model).cuda()
        model.eval()  # Set to evaluation mode


    elif model_name == 'vbsl50k_colorbg_AppleMesh00_AppleMesh01_no2dtransform_initial_backbone':
        checkpoint_path = 'vbsl50k_colorbg_AppleMesh00_AppleMesh01_no2dtransform_initial_backbone.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        # Load pre-trained ResNet-50 backbone from Barlow Twins repository
        model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        print("Loaded pretrained Barlow Twins backbone.")
        # Remove "backbone." prefix if necessary (since it was saved inside BarlowTwins model)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        backbone_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
        # Load the backbone weights
        model.load_state_dict(backbone_state_dict, strict=False)
        print("Restored weights from saved initial backbone checkpoint.")
        # Replace FC layer for binary classification (if needed)
        model.fc = nn.Linear(in_features=2048, out_features=2)
        print("Replaced final fully connected layer for binary classification.")
        # Wrap in DataParallel and move to GPU
        model = nn.DataParallel(model).cuda()
        model.eval()  # Set model to evaluation mode

    # # finetuning steps
    # elif 'SL' in model_name and 'step' in model_name:
    #     # init model
    #     model = models.resnet50(pretrained=False)
    #     model.fc = nn.Linear(2048, 2)
    #     model = nn.DataParallel(model).cuda()
    #     # load model
    #     dataset = model_name.split("_")[0]
    #     if dataset == 'vbsl50k':
    #         dataset = 'vbsl_50k'
    #     elif dataset == 'vbsl50kobj':
    #         dataset = 'vbsl_50k_obj'
    #     else:
    #         assert False
    #     step = model_name.split("_")[-1][4:]
    #     filename = f'/mnt/smb/locker/issa-locker/users/Seojin/data/saved_models/finetuning_steps/resnet50_finetune_{dataset}_seed7_step{step}_checkpoint.pth.tar'
    #     checkpoint = torch.load(filename)
    #     print(f"loaded {filename}, epoch{checkpoint['epoch']}") 
    #     model.load_state_dict(checkpoint['state_dict'])
    #     # roll back from DataParallel
    #     model = model.module
    elif 'off_the_shelf' in model_name: 
        model = models.resnet50(pretrained=False)
        if '28way' in model_name :
            model.fc = nn.Linear(2048, 28)
        elif '32way' in model_name :
            model.fc = nn.Linear(2048, 32)
        elif '24way' in model_name :
            model.fc = nn.Linear(2048, 24)
        elif '56way' in model_name :
            model.fc = nn.Linear(2048, 56)
        elif '42way' in model_name :
            model.fc = nn.Linear(2048, 42)
        elif '16way' in model_name :
            model.fc = nn.Linear(2048, 16)
        elif '14way' in model_name :
            model.fc = nn.Linear(2048, 14)
        elif '6way' in model_name :
            model.fc = nn.Linear(2048, 6)
        elif '8way' in model_name :
            model.fc = nn.Linear(2048, 8)
        elif '7way' in model_name :
            model.fc = nn.Linear(2048, 7)
        elif '10way' in model_name :
            model.fc = nn.Linear(2048, 10)
        elif '12way' in model_name :
            model.fc = nn.Linear(2048, 12)
        elif '4way' in model_name :
            model.fc = nn.Linear(2048, 4)
        else :
            model.fc = nn.Linear(2048, 2)
        model = nn.DataParallel(model).cuda()
        if model_name == 'off_the_shelf_barlowtwins_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL': # AppleMesh00-AppleMesh04
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL.pth.tar')
            print("loaded off_the_shelf_barlowtwins_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL")

        elif model_name == 'off_the_shelf_barlowtwins_finetune_12way_6ID_2EM_IDEM_colorbg': # retrained 0927
            checkpoint = torch.load('barlowtwins_finetune_6ID_2EM_IDEM_colorbg_seed77_model_best.pth.tar')
            print("loaded off_the_shelf_barlowtwins_finetune_12way_6ID_2EM_IDEM_colorbg")
        elif model_name == 'off_the_shelf_barlowtwins_finetune_24way_6ID_4EM_IDEM_colorbg': # retrained 0927
            checkpoint = torch.load('barlowtwins_finetune_6ID_4EM_IDEM_colorbg_seed777__best.pth.tar')
            print("loaded off_the_shelf_barlowtwins_finetune_24way_6ID_4EM_IDEM_colorbg")
        elif model_name == 'off_the_shelf_barlowtwins_finetune_16way_8ID_2EM_IDEM_colorbg': # retrained 0927
            checkpoint = torch.load('barlowtwins_finetune_8ID_2EM_IDEM_colorbg_seed777__best.pth.tar')
            print("loaded off_the_shelf_barlowtwins_finetune_16way_8ID_2EM_IDEM_colorbg")
        elif model_name == 'off_the_shelf_barlowtwins_finetune_32way_8ID_4EM_IDEM_colorbg': # retrained 0927
            checkpoint = torch.load('barlowtwins_finetune_8ID_4EM_IDEM_colorbg_seed777__best.pth.tar')
            print("loaded off_the_shelf_barlowtwins_finetune_32way_8ID_4EM_IDEM_colorbg")

        elif model_name == 'off_the_shelf_barlowtwins_finetune_texture_colorbg_em_neutral_anger': # retrained 0927
            checkpoint = torch.load('barlowtwins_finetune_texture_colorbg_em_neutral_anger_seed777_model_best.pth.tar')
            print("loaded off_the_shelf_barlowtwins_finetune_texture_colorbg_em_neutral_anger")
        elif model_name == 'off_the_shelf_barlowtwins_finetune_texture_colorbg_em_4way_NHAS': # retrained 0927
            checkpoint = torch.load('barlowtwins_finetune_texture_colorbg_em_4way_NHAS_seed77_model_best.pth.tar')
            print("loaded off_the_shelf_barlowtwins_finetune_texture_colorbg_em_4way_NHAS")

        elif model_name == 'off_the_shelf_barlowtwins_AppleMesh02_AppleMesh04_epochs50':
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_AppleMesh02_AppleMesh04_seed777_model_best.pth.tar')
            print(f"loaded barlowtwins_finetune_vbsle_50k_AppleMesh02_AppleMesh04_seed777_model_best_SL.pth.tar") 

        elif model_name == 'off_the_shelf_barlowtwins_texture_colorbg_2way_AppleMesh02_AppleMesh04':
            checkpoint = torch.load('barlowtwins_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_texture_colorbg_2way_AppleMesh02_AppleMesh04") 

        elif model_name == 'off_the_shelf_barlowtwins_texture_colorbg_2way_AppleMesh03_AppleMesh02':
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_AppleMesh03_AppleMesh02_colorbg_seed777_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_texture_colorbg_2way_AppleMesh03_AppleMesh02") 

        elif model_name == 'off_the_shelf_barlowtwins_texture_colorbg_2way_AppleMesh03_AppleMesh08':
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_AppleMesh03_AppleMesh08_colorbg_seed777_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_texture_colorbg_2way_AppleMesh03_AppleMesh08") 

        elif model_name == 'off_the_shelf_barlowtwins_texture_colorbg_4way':
            checkpoint = torch.load('barlowtwins_finetune_texture_colorbg_4way_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_texture_colorbg_4way") 

        elif model_name == 'off_the_shelf_barlowtwins_2way_AppleMesh03_AppleMesh08':
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_AppleMesh03_AppleMesh08_seed777_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_2way_AppleMesh03_AppleMesh08") 
        elif model_name == 'off_the_shelf_barlowtwins_2way_AppleMesh02_AppleMesh03':
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_AppleMesh02_AppleMesh03_seed777_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_2way_AppleMesh02_AppleMesh03") 

        elif model_name == 'off_the_shelf_barlowtwins_ID_8way' :
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_8way_far_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_ID_8way") 
        elif model_name == 'off_the_shelf_barlowtwins_ID_8way_epochs10' :
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_8way_far_seed77_model_best_epochs10.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_ID_8way_epochs10") 
        elif model_name == 'off_the_shelf_barlowtwins_ID_8way_epochs50' :
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_8way_far_seed77_model_best_epochs50.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_ID_8way_epochs50") 
        elif model_name == 'off_the_shelf_barlowtwins_ID_8way_colorbg_epochs50' :
            checkpoint = torch.load('barlowtwins_finetune_texture_colorbg_8way_seed77_model_best_epochs50.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_ID_8way_colorbg_epochs50") 

        elif model_name == 'off_the_shelf_barlowtwins_finetune_4way':
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_4way_KDAS_seed77_model_best.pth.tar')
            print(f"loaded barlowtwins_finetune_vbsle_50k_4way_KDAS_seed77_model_best.pth.tar") 

        elif model_name == 'off_the_shelf_barlowtwins_finetune_4way_SSKD':
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_4way_SSKD_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_4way_SSKD") 

        elif model_name == 'off_the_shelf_barlowtwins_finetune_6way_final_colorbg':
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_6way_final_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_6way_final_colorbg")
        elif model_name == 'off_the_shelf_barlowtwins_finetune_vbsle_50k_6way_final_one':
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_6way_final_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_vbsle_50k_6way_final_one")
        elif model_name == 'off_the_shelf_barlowtwins_finetune_vbsle_50k_6way_final_seed77_model_best_ver2':
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_6way_final_seed77_model_best_ver2.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_vbsle_50k_6way_final_seed77_model_best_ver2")
            
        elif model_name == 'off_the_shelf_barlowtwins_finetune_6way':
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_6way_far_seed77_model_best.pth.tar')
            print(f"loaded barlowtwins_finetune_vbsle_50k_6way_far_seed77_model_best.pth.tar") 
        # elif 'off_the_shelf_barlowtwins_finetune_8way' in model_name: # old one 
        #     checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_8way_far_seed77_model_best_8way.pth.tar')
        #     print(f"loaded barlowtwins_finetune_vbsle_50k_8way_far_seed77_model_best_8way.pth.tar") 

        elif model_name == 'off_the_shelf_barlowtwins_finetune_texture_colorbg_em_neutral_anger':
            checkpoint = torch.load('barlowtwins_finetune_texture_colorbg_em_neutral_anger_seed777_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_texture_colorbg_em_neutral_anger") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_texture_colorbg_em_4way_NHAS':
            checkpoint = torch.load('barlowtwins_finetune_texture_colorbg_em_4way_NHAS_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_texture_colorbg_em_4way_NHAS") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_texture_colorbg_em_6way_excl_fear':
            checkpoint = torch.load('barlowtwins_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_texture_colorbg_em_6way_excl_fear") 

        elif model_name == 'off_the_shelf_barlowtwins_finetune_10way':
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_10way_seed77_model_best_10way.pth.tar')
            print(f"loaded barlowtwins_finetune_vbsle_50k_10way_seed77_model_best_10way.pth.tar") 
        elif   model_name == 'off_the_shelf_barlowtwins_finetune_12way':
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_12way_seed77_model_best_12way.pth.tar')
            print(f"loaded barlowtwins_finetune_vbsle_50k_12way_seed77_model_best_12way.pth.tar") 
        elif model_name == 'off_the_shelf_barlowtwins_IDEM_14way' :
            checkpoint = torch.load('barlowtwins_finetune_14way_EM_AppleMesh02_AppleMesh03_seed777_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_IDEM_14way") 
        elif model_name == 'off_the_shelf_barlowtwins_IDEM_14way_colorbg' :
            checkpoint = torch.load('barlowtwins_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_IDEM_14way_colorbg") 
        elif model_name == 'off_the_shelf_barlowtwins_IDEM_14way_epochs10' :
            checkpoint = torch.load('barlowtwins_finetune_14way_EM_AppleMesh02_AppleMesh03_seed777_model_best_epochs10.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_IDEM_14way_epochs10") 
        elif model_name == 'off_the_shelf_barlowtwins_IDEM_14way_epochs50' :
            checkpoint = torch.load('barlowtwins_finetune_14way_EM_AppleMesh02_AppleMesh03_seed777_model_best_epochs50.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_IDEM_14way_epochs50") 
        elif model_name == 'off_the_shelf_barlowtwins_IDEM_14way_colorbg_epochs50' :
            checkpoint = torch.load('barlowtwins_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_epochs50.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_IDEM_14way_epochs50") 
        elif model_name == 'off_the_shelf_barlowtwins_IDEM_14way_graybg_epochs50' :
            checkpoint = torch.load('barlowtwins_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_grayscaled_epochs50.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_IDEM_14way_graybg_epochs50") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_14way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed777_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_14way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed777_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_14way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed777_model_best") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_14way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed777_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_14way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed777_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_14way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed777_model_best") 

        elif model_name == 'off_the_shelf_barlowtwins_IDEM_28way_epochs50' :
            checkpoint = torch.load('barlowtwins_finetune_28way_IDEM_seed777_model_best_epoch50.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_IDEM_28way_epochs50") 
        elif model_name == 'off_the_shelf_barlowtwins_IDEM_28way_colorbg_epochs50' :
            checkpoint = torch.load('barlowtwins_finetune_28way_IDEM_colorbg_seed777_model_best_epoch50.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_IDEM_28way_colorbg_epochs50") 
        # elif model_name == 'off_the_shelf_barlowtwins_IDEM_28way_colorbg_epochs50' :
        #     checkpoint = torch.load('barlowtwins_finetune_28way_IDEM_colorbg_seed777_model_best_epoch50.pth.tar')
        #     print(f"loaded off_the_shelf_barlowtwins_IDEM_28way_colorbg_epochs50") 
        elif model_name == 'off_the_shelf_barlowtwins_IDEM_28way_graybg_epochs50' :
            checkpoint = torch.load('barlowtwins_finetune_28way_IDEM_colorbg_seed777_model_best_grayscaled_epoch50.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_IDEM_28way_graybg_epochs50") 

        elif model_name == 'off_the_shelf_barlowtwins_finetune_4way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_4way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_4way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_4way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_4way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_4way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_4way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_4way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_4way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best") 

        elif model_name == 'off_the_shelf_barlowtwins_finetune_6way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_6way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_6way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_6way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_6way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_6way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_6way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_6way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_6way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best") 

        elif model_name == 'off_the_shelf_barlowtwins_finetune_8way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_8way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_8way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_8way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_8way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_8way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_8way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best' :
            checkpoint = torch.load('off_the_shelfbarlowtwins_finetune_8way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_epochs50.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_IDEM_8way_colorbg_epochs50") 

        elif model_name == 'off_the_shelf_barlowtwins_finetune_10way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_10way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_10way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_10way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_10way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_10way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_10way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_10way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_10way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best") 

        elif model_name == 'off_the_shelf_barlowtwins_finetune_12way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_12way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_12way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_12way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_12way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_12way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_12way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_12way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_12way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best") 

        elif model_name == 'off_the_shelf_barlowtwins_finetune_8way_IDEM_ssskd_colorbg' :
            checkpoint = torch.load('barlowtwins_finetune_8way_IDEM_ssskd_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_8way_IDEM_ssskd_colorbg") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_16way_IDEM_ssskd_colorbg' :
            checkpoint = torch.load('barlowtwins_finetune_16way_IDEM_ssskd_colorbg_seed777_.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_16way_IDEM_ssskd_colorbg") 

        elif model_name == 'off_the_shelf_barlowtwins_IDEM_16way_colorbg_epochs50' :
            checkpoint = torch.load('off_the_shelfbarlowtwins_finetune_16way_IDEM_seed777_model_best_epochs_50.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_IDEM_16way_colorbg_epochs50") 
        
        elif model_name == 'off_the_shelf_barlowtwins_finetune_56way_IDEM_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_56way_IDEM_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_56way_IDEM_colorbg_seed77_model_best") 

        elif model_name == 'off_the_shelf_barlowtwins_finetune_42way_IDEM_colorbg_seed777_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_42way_IDEM_colorbg_seed777_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_42way_IDEM_colorbg_seed777_model_best") 

        elif model_name == 'off_the_shelf_barlowtwins_finetune_7way_EM_AppleMesh02_seed77_texture_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_7way_EM_AppleMesh02_seed77__texture_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_7way_EM_AppleMesh02_seed77_texture_model_best") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_7way_EM_AppleMesh03_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_7way_EM_AppleMesh03_colorbg_seed77__model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_7way_EM_AppleMesh03_colorbg_seed77_model_best") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_7way_EM_AppleMesh04_colorbg_seed77__model_best' :
            checkpoint = torch.load('barlowtwins_finetune_7way_EM_AppleMesh04_colorbg_seed77__model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_7way_EM_AppleMesh04_colorbg_seed77__model_best") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_7way_EM_AppleMesh08_colorbg_seed77__model_best' :
            checkpoint = torch.load('barlowtwins_finetune_7way_EM_AppleMesh08_colorbg_seed77__model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_7way_EM_AppleMesh08_colorbg_seed77__model_best") 

        elif model_name == 'off_the_shelf_barlowtwins_finetune_vbsle_50k_4way_SSKD_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_4way_SSKD_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_vbsle_50k_4way_SSKD_seed77_model_best") 
        elif model_name == 'off_the_shelf_barlowtwins_finetune_vbsle_50k_4way_SSKD_colorbg_seed77_model_best' :
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_4way_SSKD_colorbg_seed77_model_best.pth.tar')
            print(f"loaded off_the_shelf_barlowtwins_finetune_vbsle_50k_4way_SSKD_colorbg_seed77_model_best") 
        print("Before loading:", model.module.fc.weight.shape)

        model.load_state_dict(checkpoint['state_dict'])
        print("After loading:", model.module.fc.weight.shape)

        # Roll back from DataParallel to return the plain model
        model = model.module
        print(f"Model loaded successfully: {model_name}")

   
    elif model_name == '14way_EM_AppleMesh02_AppleMesh03_barlowtwins_pretrained2_final_model':
        checkpoint_path = '/mnt/smb/locker/issa-locker/users/Seojin/data/saved_models/14way_EM_AppleMesh02_AppleMesh03_barlowtwins_pretrained3_final_model_proj.pth.tar'
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint  # Adjust based on your checkpoint structure
        model = BarlowTwins(args)
        backbone_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
        model.backbone.load_state_dict(backbone_state_dict, strict=False)
        projector_state_dict = {k.replace("projector.", ""): v for k, v in state_dict.items() if k.startswith("projector.")}
        # model.projector.load_state_dict(projector_state_dict, strict=False)

        # Print before loading weights
        print("Before loading projector weights:", model.projector[0].weight[0, :5])

        # Load the projector weights
        model.projector.load_state_dict(projector_state_dict, strict=False)

        # Print after loading weights
        print("After loading projector weights:", model.projector[0].weight[0, :5])


        model = model.cuda()
        model.eval()  # Set model to evaluation mode 
    
    elif model_name == 'SL_imagenet_pretrained_AppleMesh00_AppleMesh01_bt':
        checkpoint_path = '/mnt/smb/locker/issa-locker/users/Seojin/data/saved_models/AppleMesh00_AppleMesh01_50k_best_model.pth.tar'
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint  # Adjust based on your checkpoint structure
        model = BarlowTwins(args)
        model.load_state_dict(state_dict, strict=False)
        model = model.cuda()
        model.eval()  # Set model to evaluation mode 
        
    elif 'bt' in model_name or 'barlowtwins' in model_name:
        # Initialize a ResNet50 backbone for Barlow Twins
        model = models.resnet50(pretrained=False)
        # model.fc = nn.Identity()  # Set fc as Identity to match Barlow Twins architecture
        

        # Load the specific Barlow Twins model
        if model_name == 'SL_barlowtwins_pretrained_AppleMesh00_AppleMesh01_bt':
            checkpoint_path = '/mnt/smb/locker/issa-locker/users/Seojin/data/saved_models/AppleMesh00_AppleMesh01_50k_barlowtwins_pretrained_best_model.pth'
            print(f"Loading model from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path)

            # Remove 'module.' prefix if present
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.module.load_state_dict(state_dict, strict=False)


        elif model_name == '14way_EM_AppleMesh02_AppleMesh03_barlowtwins_pretrained_final_model':
            model.fc = nn.Linear(2048, 14)  # Set fc as Identity to match Barlow Twins architecture
            # model = nn.DataParallel(model).cuda()
            checkpoint = torch.load('/mnt/smb/locker/issa-locker/users/Seojin/data/saved_models/14way_EM_AppleMesh02_AppleMesh03_barlowtwins_pretrained_final_model.pth.tar')
            print(checkpoint.keys())
            model.load_state_dict(checkpoint, strict=False)
            model = nn.DataParallel(model).cuda()

        elif model_name == '14way_EM_AppleMesh02_AppleMesh03_barlowtwins_pretrained_best_model':
            model.fc = nn.Linear(2048, 14)  # Set fc as Identity to match Barlow Twins architecture
            model = nn.DataParallel(model).cuda()
            checkpoint = torch.load('/mnt/smb/locker/issa-locker/users/Seojin/data/saved_models/14way_EM_AppleMesh02_AppleMesh03_barlowtwins_pretrained_best_model.pth.tar')
            print(checkpoint.keys())
            model.load_state_dict(checkpoint, strict=False)
        



        elif model_name == 'barlowtwins' :
            # Load pre-trained ResNet-50 backbone from Barlow Twins repository
            pretrained_backbone = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
            print("Loaded pretrained Barlow Twins backbone.")
            pretrained_backbone.fc = nn.Linear(in_features=2048, out_features=2)
            print("Replaced final fully connected layer for binary classification.")
            model = nn.DataParallel(pretrained_backbone).cuda()
            checkpoint_path = '/mnt/smb/locker/issa-locker/users/Seojin/saved_models/checkpoint.pth'  # Replace with the actual path to your checkpoint file
            checkpoint = torch.load(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
            model.load_state_dict(checkpoint['model'], strict=False)
            print("Loaded model state dictionary from checkpoint.")
        elif model_name == 'barlowtwins_layer1' :
            # Load pre-trained ResNet-50 backbone from Barlow Twins repository
            pretrained_backbone = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
            print("Loaded pretrained Barlow Twins backbone.")
            pretrained_backbone.fc = nn.Linear(in_features=2048, out_features=2)
            print("Replaced final fully connected layer for binary classification.")
            model = nn.DataParallel(pretrained_backbone).cuda()
            checkpoint_path = '/mnt/smb/locker/issa-locker/users/Seojin/saved_models/checkpoint.pth'  # Replace with the actual path to your checkpoint file
            checkpoint = torch.load(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
            model.load_state_dict(checkpoint['model'], strict=False)
            print("Loaded model state dictionary from checkpoint.")
        elif model_name == 'barlowtwins_layer2' :
            # Load pre-trained ResNet-50 backbone from Barlow Twins repository
            pretrained_backbone = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
            print("Loaded pretrained Barlow Twins backbone.")
            pretrained_backbone.fc = nn.Linear(in_features=2048, out_features=2)
            print("Replaced final fully connected layer for binary classification.")
            model = nn.DataParallel(pretrained_backbone).cuda()
            checkpoint_path = '/mnt/smb/locker/issa-locker/users/Seojin/saved_models/checkpoint.pth'  # Replace with the actual path to your checkpoint file
            checkpoint = torch.load(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
            model.load_state_dict(checkpoint['model'], strict=False)
            print("Loaded model state dictionary from checkpoint.")
        elif model_name == 'barlowtwins_layer3' :
            # Load pre-trained ResNet-50 backbone from Barlow Twins repository
            pretrained_backbone = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
            print("Loaded pretrained Barlow Twins backbone.")
            pretrained_backbone.fc = nn.Linear(in_features=2048, out_features=2)
            print("Replaced final fully connected layer for binary classification.")
            model = nn.DataParallel(pretrained_backbone).cuda()
            checkpoint_path = '/mnt/smb/locker/issa-locker/users/Seojin/saved_models/checkpoint.pth'  # Replace with the actual path to your checkpoint file
            checkpoint = torch.load(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
            model.load_state_dict(checkpoint['model'], strict=False)
            print("Loaded model state dictionary from checkpoint.")
        elif model_name == 'barlowtwins_layer4_no_pooling' :
            # Load pre-trained ResNet-50 backbone from Barlow Twins repository
            pretrained_backbone = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
            print("Loaded pretrained Barlow Twins backbone.")
            pretrained_backbone.fc = nn.Linear(in_features=2048, out_features=2)
            print("Replaced final fully connected layer for binary classification.")
            model = nn.DataParallel(pretrained_backbone).cuda()
            checkpoint_path = '/mnt/smb/locker/issa-locker/users/Seojin/saved_models/checkpoint.pth'  # Replace with the actual path to your checkpoint file
            checkpoint = torch.load(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
            model.load_state_dict(checkpoint['model'], strict=False)
            print("Loaded model state dictionary from checkpoint.")

        model = model.module
    elif 'head' in model_name :
        if '14way' in model_name : 
            num_ids = 14
        model = MultiHeadResNet(num_classes1=7, num_classes2=7)
        model = nn.DataParallel(model).cuda()
        if model_name == 'SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head1' :
            checkpoint = torch.load('resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head1.pth')
            print("loaded resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head1")
        elif model_name == 'SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head2' :
            checkpoint = torch.load('resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head2.pth')
            print("loaded resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head2")
        
        elif model_name == 'SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_heads_combined' :
            checkpoint = torch.load('resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_heads_combined.pth')
            print("loaded resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_heads_combined")
        elif model_name == 'SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head1_epoch2' :
            checkpoint = torch.load('resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head1_epoch2.pth')
            print("loaded resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head1_epoch2")
        elif model_name == 'SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head2_epoch2' :
            checkpoint = torch.load('resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head2_epoch2.pth')
            print("loaded resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head2_epoch2")
        elif model_name == 'SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_heads_combined_epoch2' :
            checkpoint = torch.load('resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_heads_combined_epoch2.pth')
            print("loaded resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_heads_combined_epoch2")
        elif model_name == 'SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head1_epoch10' :
            checkpoint = torch.load('resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head1_epoch10.pth')
            print("loaded resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head1_epoch10")
        elif model_name == 'SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head2_epoch10' :
            checkpoint = torch.load('resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head2_epoch10.pth')
            print("loaded resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head2_epoch10")
        elif model_name == 'SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_heads_combined_epoch10' :
            checkpoint = torch.load('resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_heads_combined_epoch10.pth')
            print("loaded resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_heads_combined_epoch10")

        elif model_name == 'SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head1_alternate' :
            checkpoint = torch.load('resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head1_alternate.pth')
            print("loaded resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head1_alternate")
        elif model_name == 'SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head2_alternate' :
            checkpoint = torch.load('resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head2_alternate.pth')
            print("loaded resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_head2_alternate")
        elif model_name == 'SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_heads_combined_alternate' :
            checkpoint = torch.load('resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_heads_combined_alternate.pth')
            print("loaded resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_best_heads_combined_epoch10")

        
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.backbone.'):
                new_key = k.replace('module.backbone.', 'module.backbone.')  # Keep backbone keys unchanged
            elif k.startswith('module.fc1.') and 'head1' in model_name:
                new_key = k.replace('module.fc1.', 'module.fc1.')  # Load for head1
            elif k.startswith('module.fc2.') and 'head2' in model_name:
                new_key = k.replace('module.fc2.', 'module.fc2.')  # Load for head2
            else:
                new_key = k  # Keep other keys unchanged
            new_state_dict[new_key] = v

        # Load modified state_dict into the model
        model.load_state_dict(new_state_dict, strict=False)
    elif model_name == 'SL_resnet50_finetune_remapping_best' :
        num_classes = 7  # Set default number of classes
        model = SharedEmotionResNet(num_classes=num_classes)  # Initialize model
        model = nn.DataParallel(model).cuda()  # Ensure DataParallel if using multiple GPUs
        checkpoint = torch.load('resnet50_finetune_shared_emotion_best.pth')
        print("loaded SL_resnet50_finetune_shared_emotion_best")
        model.module.fc = nn.Linear(2048, num_classes).cuda()
        new_state_dict = {}
        for k, v in checkpoint.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v

        # Load adjusted state dict
        model.load_state_dict(new_state_dict, strict=False)


    elif 'ova' in model_name:
        num_classes = 7
        model = OneVsAllResNet(num_classes=num_classes)  
        model = nn.DataParallel(model).cuda()  

        # Mapping model names to class indices
        class_idx_map = {
            'SL_resnet50_finetune_ova_AppleMesh02_anger': 0,
            'SL_resnet50_finetune_ova_AppleMesh02_disgust': 1,
            'SL_resnet50_finetune_ova_AppleMesh02_fear': 2,
            'SL_resnet50_finetune_ova_AppleMesh02_happiness': 3,
            'SL_resnet50_finetune_ova_AppleMesh02_neutral': 4,
            'SL_resnet50_finetune_ova_AppleMesh02_sadness': 5,
            'SL_resnet50_finetune_ova_AppleMesh02_surprise': 6
        }

        if model_name in class_idx_map:
            checkpoint_path = f"resnet50_finetune_ova_AppleMesh02_{model_name.split('_')[-1]}.pth"
            checkpoint = torch.load(checkpoint_path)
            model.module.set_class_idx(class_idx_map[model_name])  # ✅ Set class_idx in the model
            print(f"Loaded {model_name}, assigned class_idx: {model.module.class_idx}")

        # Load adjusted state dict
        new_state_dict = {k.replace('module.', '') if k.startswith('module.') else k: v for k, v in checkpoint.items()}
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        if missing:
            print(f"Warning: Missing keys in checkpoint: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected}")

        model.eval()

    elif model_name == 'SL_basel_colortexture_1kid_finetune_vbsle_50k_AppleMesh02_AppleMesh04_seed777_model_best' :
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 1000)
        model = nn.DataParallel(model).cuda()
        # load model (2way emotion)
        checkpoint = torch.load('basel_colortexture_1kid_finetune_vbsle_50k_AppleMesh02_AppleMesh04_seed777_model_best.pth.tar')
        print(f"loaded SL_basel_colortexture_1kid_finetune_vbsle_50k_AppleMesh02_AppleMesh04_seed777_model_best") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel`
        model = model.module

    elif model_name == 'SL_basel_colortexture_1kid_finetune_56way_IDEM_colorbg_seed777_' :
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 1000)
        model = nn.DataParallel(model).cuda()
        # load model (2way emotion)
        checkpoint = torch.load('basel_colortexture_1kid_finetune_56way_IDEM_colorbg_seed777_.pth.tar')
        print(f"loaded SL_basel_colortexture_1kid_finetune_56way_IDEM_colorbg_seed777_") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel`
        model = model.module

    elif 'neutral' in model_name or 'em' in model_name or 'EM' in model_name or 'happiness' in model_name or 'anger' in model_name: 
        if '28way' in model_name : 
            num_ids = 28
        elif '24way' in model_name : 
            num_ids = 24
        elif '32way' in model_name : 
            num_ids = 32
        elif '16way' in model_name : 
            num_ids = 16
        elif '56way' in model_name : 
            num_ids =56
        elif '42way' in model_name : 
            num_ids =42
        elif '14way' in model_name : 
            num_ids = 14
        elif '12way' in model_name :
            num_ids = 12
        elif '10way' in model_name :
            num_ids = 10
        elif '8way' in model_name :
            num_ids = 8
        elif '6way' in model_name :
            num_ids = 6
        elif '7way' in model_name : 
            num_ids = 7
        elif '4way' in model_name :
            num_ids = 4
        else:
            num_ids = 2
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_ids)
        model = nn.DataParallel(model).cuda()
        # load model (2way emotion)
        if model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_neutral_happiness_seed777_model_best_SL' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_neutral_happiness_seed777_model_best_SL.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_neutral_happiness_seed777_model_best_SL") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_neutral_sadness_seed777_model_best_SL' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_neutral_sadness_seed777_model_best_SL.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_neutral_sadness_seed777_model_best_SL") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_neutral_disgust_seed777_model_best_SL' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_neutral_disgust_seed777_model_best_SL.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_neutral_disgust_seed777_model_best_SL") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_neutral_anger_seed777_model_best_SL' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_neutral_anger_seed777_model_best_SL.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_neutral_anger_seed777_model_best_SL") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_happiness_sadness_seed777_model_best_SL' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_happiness_sadness_seed777_model_best_SL.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_happiness_sadness_seed777_model_best_SL") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_fear_anger_seed777_model_best_SL' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_fear_anger_seed777_model_best_SL.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_fear_anger_seed777_model_best_SL") 
        # load model (4way emotion)
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_em_4way_NFDS_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_em_4way_NFDS_seed77_model_best.pth.tar')
            print(f"loaded resnet50_finetune_vbsle_50k_em_4way_NFDS_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_em_4way_NHSF_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_em_4way_NHSF_seed77_model_best.pth.tar')
            print(f"loaded resnet50_finetune_vbsle_50k_em_4way_NHSF_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_em_4way_HASF_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_em_4way_HASF_seed77_model_best.pth.tar')
            print(f"loaded resnet50_finetune_vbsle_50k_em_4way_HASF_seed77_model_best") 
        # load model (6way emotion)
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_em_6way_excl_happiness_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_em_6way_excl_happiness_seed77_model_best.pth.tar')
            print(f"loaded resnet50_finetune_vbsle_50k_em_6way_excl_happiness_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_em_6way_excl_surprise_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_em_6way_excl_surprise_seed77_model_best.pth.tar')
            print(f"loaded resnet50_finetune_vbsle_50k_em_6way_excl_surprise_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_em_6way_excl_neutral_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_em_6way_excl_neutral_seed77_model_best.pth.tar')
            print(f"loaded resnet50_finetune_vbsle_50k_em_6way_excl_neutral_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_neutral_disgust': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_neutral_disgust_seed777_model_best_SL.pth.tar')
            print("loaded resnet50_finetune_vbsle_50k_neutral_disgust_seed777_model_best_SL")

        elif model_name == 'SL_resnet50_finetune_vbsle_50k_neutral_surprise': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_neutral_surprise_seed777_model_best_SL.pth.tar')
            print("loaded resnet50_finetune_vbsle_50k_neutral_surprise_seed_777_model_best_SL")

        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh03_emotion_mix_seed777_model_best_SL': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh03_emotion_mix_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh03_emotion_mix_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh03_emotion_full_seed777_model_best_SL': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh03_emotion_full_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh03_emotion_full_seed777_model_best_SL")

        elif model_name == 'SL_resnet50_finetune_12way_6ID_2EM_IDEM_colorbg': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_6ID_2EM_IDEM_colorbg_seed777__best.pth.tar')
            print("loaded SL_resnet50_finetune_12way_6ID_2EM_IDEM_colorbg")
        elif model_name == 'SL_resnet50_finetune_24way_6ID_4EM_IDEM_colorbg': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_6ID_4EM_IDEM_colorbg_seed777_.pth.tar')
            print("loaded SL_resnet50_finetune_24way_6ID_4EM_IDEM_colorbg")
        elif model_name == 'SL_resnet50_finetune_16way_8ID_2EM_IDEM_colorbg': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_8ID_2EM_IDEM_colorbg_seed777__best.pth.tar')
            print("loaded SL_resnet50_finetune_8ID_2EM_IDEM_colorbg")
        elif model_name == 'SL_resnet50_finetune_32way_8ID_4EM_IDEM_colorbg': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_8ID_4EM_IDEM_colorbg_seed777_.pth.tar')
            print("loaded SL_resnet50_finetune_8ID_4EM_IDEM_colorbg")

        elif model_name == 'SL_resnet50_finetune_vbsle_50k_em_6way_excl_fear_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_em_6way_excl_fear_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_em_6way_excl_fear_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_texture_em_neutral_anger_seed777_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_texture_em_neutral_anger_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_texture_em_neutral_anger_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_texture_em_4way_NHAS_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_texture_em_4way_NHAS_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_texture_em_4way_NHAS_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_texture_em_6way_excl_fear_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_texture_em_6way_excl_fear_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_texture_em_6way_excl_fear_seed77_model_best")

    

        elif model_name == 'SL_resnet50_finetune_7way_EM_AppleMesh02_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_7way_EM_AppleMesh02_seed77__no_texture_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_7way_EM_AppleMesh02_seed77_model_best")
        
        elif model_name == 'SL_resnet50_finetune_7way_EM_AppleMesh03_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_7way_EM_AppleMesh03_colorbg_seed77__model_best.pth.tar')
            print("loaded SL_resnet50_finetune_7way_EM_AppleMesh03_colorbg_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_7way_EM_AppleMesh04_colorbg_seed77__model_best' :
            checkpoint = torch.load('resnet50_finetune_7way_EM_AppleMesh04_colorbg_seed77__model_best.pth.tar')
            print("loaded SL_resnet50_finetune_7way_EM_AppleMesh04_colorbg_seed77__model_best")
        elif model_name == 'SL_resnet50_finetune_7way_EM_AppleMesh08_colorbg_seed77__model_best' :
            checkpoint = torch.load('resnet50_finetune_7way_EM_AppleMesh08_colorbg_seed77__model_best.pth.tar')
            print("loaded SL_resnet50_finetune_7way_EM_AppleMesh08_colorbg_seed77__model_best")

        elif model_name == 'SL_colorbg_resnet50_finetune_7way_EM_AppleMesh02_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_7way_EM_AppleMesh02_seed77_model_best.pth.tar')
            print("loaded SL_colorbg_resnet50_finetune_7way_EM_AppleMesh02_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed777_model_best_epochs10' :
            checkpoint = torch.load('resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed777_model_best_epochs10.pth.tar')
            print("loaded SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed777_model_best_epochs10")
        elif model_name == 'SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed777_model_best_epochs50' :
            checkpoint = torch.load('resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed777_model_best_epochs50.pth.tar')
            print("loaded SL_resnet50_finetune_14way_EM_AppleMesh02_AppleMesh03_seed777_model_best_epochs50")

        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed777_model_best' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed777_model_best' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed777_model_best")

        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_neutral_anger_seed777_model_best' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_neutral_anger_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_neutral_anger_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_4way_NHAS_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_4way_NHAS_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_4way_NHAS_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed777_model_best_dynamicblur_linear_grayscaled_ver2' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed777_model_best_dynamicblur_linear_grayscaled_ver2.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed777_model_best_dynamicblur_linear_grayscaled_ver2")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_neutral_anger_seed777_model_best_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_neutral_anger_seed777_model_best_grayscaled.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_neutral_anger_seed777_model_best_grayscaled")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_neutral_anger_seed777_model_best_grayscaled_blurred_ver5' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_neutral_anger_seed777_model_best_grayscaled_blurred_ver5.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_neutral_anger_seed777_model_best_grayscaled_blurred_ver5")
        
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_neutral_anger_seed777_model_best_grayscaled_blurred_ver5_kernel9' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_neutral_anger_seed777_model_best_grayscaled_blurred_ver5_kernel9.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_neutral_anger_seed777_model_best_grayscaled_blurred_ver5_kernel9")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_neutral_anger_seed777_model_best_grayscaled_blurred_ver6_kernel9' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_neutral_anger_seed777_model_best_grayscaled_blurred_ver6_kernel9.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_neutral_anger_seed777_model_best_grayscaled_blurred_ver6_kernel9")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed777_model_best_dynamicblur_linear_grayscaled_ver2' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed777_model_best_dynamicblur_linear_grayscaled_ver2.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed777_model_best_dynamicblur_linear_grayscaled_ver2")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_4way_NHAS_seed77_model_best_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_4way_NHAS_seed77_model_best_grayscaled.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_4way_NHAS_seed77_model_best_grayscaled")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_grayscaled.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_grayscaled")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_grayscaled_epochs20' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_grayscaled_epochs20.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_grayscaled_epochs20")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed777_model_best_dynamicblur_nonlinear_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed777_model_best_dynamicblur_nonlinear_grayscaled.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed777_model_best_dynamicblur_nonlinear_grayscaled")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed777_model_best_dynamicblur_linear_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed777_model_best_dynamicblur_linear_grayscaled.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed777_model_best_dynamicblur_linear_grayscaled")

        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_epochs10_sigma3' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_epochs10_sigma3.pth.tar')
            print("loaded resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_epochs10_sigma3")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_grayscaled_blurred_epochs20_ver4' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_grayscaled_blurred_epochs20_ver4.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_grayscaled_blurred_epochs20_ver4")

        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_grayscaled_blurred_epochs10_sigma3_kernel9' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_grayscaled_blurred_epochs10_sigma3_kernel9.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_grayscaled_blurred_epochs10_sigma3_kernel9")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_grayscaled_blurred_epochs10_sigma2' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_grayscaled_blurred_epochs10_sigma2.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_grayscaled_blurred_epochs10_sigma2")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_epochs10_sigma5' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_epochs10_sigma5.pth.tar')
            print("loaded resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_epochs10_sigma5")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_grayscaled_epochs20_ver2' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_grayscaled_epochs20_ver2.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_grayscaled_epochs20_ver2")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_epochs20_ver2' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_epochs20_ver2.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_epochs20_ver2")
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_epochs20_ver3' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_epochs20_ver3.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_em_6way_excl_fear_seed77_model_best_blurred_epochs20_ver3")

        elif model_name == 'SL_resnet50_finetune_4way_IDEM_AppleMesh02_AppleMesh03_NA_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_4way_IDEM_AppleMesh02_AppleMesh03_NA_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_4way_IDEM_AppleMesh02_AppleMesh03_NA_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_4way_IDEM_AppleMesh02_AppleMesh03_HA_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_4way_IDEM_AppleMesh02_AppleMesh03_HA_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_4way_IDEM_AppleMesh02_AppleMesh03_HA_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_6way_IDEM_AppleMesh02_AppleMesh03_HAS_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_6way_IDEM_AppleMesh02_AppleMesh03_HAS_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_6way_IDEM_AppleMesh02_AppleMesh03_HAS_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_8way_IDEM_AppleMesh02_AppleMesh03_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_8way_IDEM_AppleMesh02_AppleMesh03_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_8way_IDEM_AppleMesh02_AppleMesh03_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_10way_IDEM_AppleMesh02_AppleMesh03_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_10way_IDEM_AppleMesh02_AppleMesh03_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_10way_IDEM_AppleMesh02_AppleMesh03_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_12way_IDEM_AppleMesh02_AppleMesh03_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_12way_IDEM_AppleMesh02_AppleMesh03_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_12way_IDEM_AppleMesh02_AppleMesh03_seed77_model_best")
        
        elif model_name == 'SL_resnet50_finetune_4way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_4way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_4way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_4way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_4way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_grayscaled.pth.tar')

        elif model_name == 'SL_resnet50_finetune_4way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_4way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_4way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_4way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_4way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_4way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best")

            print("loaded SL_resnet50_finetune_4way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_grayscaled")
        elif model_name == 'SL_resnet50_finetune_6way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best' :

            checkpoint = torch.load('resnet50_finetune_6way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_6way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_6way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_6way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_grayscaled.pth.tar')
            print("loaded SL_resnet50_finetune_6way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_grayscaled")

        elif model_name == 'SL_resnet50_finetune_6way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_6way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_6way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_6way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_6way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_6way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best")

        elif model_name == 'SL_resnet50_finetune_8way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_8way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_8way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_8way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_8way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_grayscaled.pth.tar')
            print("loaded SL_resnet50_finetune_8way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_grayscaled")

        elif model_name == 'SL_resnet50_finetune_8way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_8way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_8way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_8way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_8way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_8way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best")

        elif model_name == 'SL_resnet50_finetune_10way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_10way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_10way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_10way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_10way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best.pth.tar')
        elif model_name == 'SL_resnet50_finetune_10way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_10way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_10way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_12way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_12way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_12way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best")
            print("loaded SL_resnet50_finetune_10way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_12way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_12way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_12way_IDEM_AppleMesh02_AppleMesh04_colorbg_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_12way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_12way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_12way_IDEM_AppleMesh03_AppleMesh08_colorbg_seed77_model_best")


        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_epoch4' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_epoch4.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_epoch4")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_grayscaled_epoch4' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_grayscaled_epoch4.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_grayscaled_epoch4")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_grayscaled.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best_grayscaled")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_20250303' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_20250303.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_20250303")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_grayscaled_20250303' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_grayscaled_20250303.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_grayscaled_20250303")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_grayscaled.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_grayscaled")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_grayscaled.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_grayscaled")

        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_sigma05' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_sigma05.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_sigma05")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_sigma15' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_sigma15.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_sigma15")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_sigma3' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_sigma3.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_sigma3")

        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_sigma15_ver2' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_sigma15_ver2.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_sigma15_ver2")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_epochs4' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_epochs4.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_epochs4")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_epochs10' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_epochs10.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_epochs10")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_epochs50' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_epochs50.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_blurred_epochs50")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best2' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best2.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed77_model_best2")
        
        # 28way
        elif model_name == 'SL_resnet50_finetune_28way_IDEM_colorbg_seed777_model_best' :
            checkpoint = torch.load('resnet50_finetune_28way_IDEM_colorbg_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_28way_IDEM_colorbg_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_28way_IDEM_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_28way_IDEM_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_28way_IDEM_colorbg_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_28way_IDEM_colorbg_seed777_model_best_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_28way_IDEM_colorbg_seed777_model_best_grayscaled.pth.tar')
            print("loaded SL_resnet50_finetune_28way_IDEM_colorbg_seed777_model_best_grayscaled")
        elif model_name == 'SL_resnet50_finetune_28way_IDEM_seed777_model_best' :
            checkpoint = torch.load('resnet50_finetune_28way_IDEM_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_28way_IDEM_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_28way_IDEM_colorbg_seed777_model_best_250430' :
            checkpoint = torch.load('resnet50_finetune_28way_IDEM_colorbg_seed777_model_best_250430.pth.tar')
            print("loaded SL_resnet50_finetune_28way_IDEM_colorbg_seed777_model_best_250430")
        # 42 way
        elif model_name == 'SL_resnet50_finetune_42way_IDEM_colorbg_seed777_model_best' :
            checkpoint = torch.load('resnet50_finetune_42way_IDEM_colorbg_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_42ay_IDEM_colorbg_seed777_model_best")
        # 56 way
        elif model_name == 'SL_resnet50_finetune_56way_IDEM_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_56way_IDEM_colorbg_seed77_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_56way_IDEM_colorbg_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_56way_IDEM_colorbg_seed777__from_scratch' :
            checkpoint = torch.load('resnet50_finetune_56way_IDEM_colorbg_seed777__from_scratch.pth.tar')
            print("loaded SL_resnet50_finetune_56way_IDEM_colorbg_seed777__from_scratch")
        # Dynamic blur
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_dynamicblur_linear' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_dynamicblur_linear.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_dynamicblur_linear")
        elif model_name == 'SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_dynamicblur_linear_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_dynamicblur_linear_grayscaled.pth.tar')
            print("loaded SL_resnet50_finetune_14way_IDEM_AppleMesh02_AppleMesh03_colorbg_seed777_model_best_dynamicblur_linear_grayscaled")
        elif model_name == 'SL_resnet50_finetune_16way_IDEM_ssskd_colorbg_seed777' :
            checkpoint = torch.load('resnet50_finetune_16way_IDEM_ssskd_colorbg_seed777_.pth.tar')
            print(f"loaded SL_resnet50_finetune_16way_IDEM_ssskd_colorbg_seed777") 
        elif model_name == 'SL_resnet50_finetune_8way_IDEM_ssskd_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_8way_IDEM_ssskd_colorbg_seed77_model_best.pth.tar')
            print(f"loaded SL_resnet50_finetune_8way_IDEM_ssskd_colorbg_seed77_model_best") 

        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel`
        model = model.module


    elif 'untrained' in model_name: 
        if model_name == 'SL_resnet_untrained_camel_elephant_finetune_vbsle_50k_6way_close_seed77_model_best_4way' :
            num_ids = 6
        elif model_name == 'SL_resnet_untrained_camel_elephant_finetune_vbsle_50k_10way_seed77_model_best_10way' :
            num_ids = 10
        elif model_name == 'SL_resnet_untrained_camel_elephant_finetune_vbsle_50k_12way_seed77_model_best_12way' :
            num_ids = 12
        else:
            num_ids = 2
            # init model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_ids)
        model = nn.DataParallel(model).cuda()
            # load model
        if model_name == 'SL_resnet_untrained_camel_elephant_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL' :
            checkpoint = torch.load('resnet_untrained_camel_elephant_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL.pth.tar')
            print(f"loaded resnet_untrained_camel_elephant_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_camel_elephant_seed777_model_best_SL_untrained' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_camel_elephant_seed777_model_best_SL_untrained.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_camel_elephant_seed777_model_best_SL_untrained") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL_untrained' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL_untrained.pth.tar')
            print(f"loaded resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL_untrained") 
        elif model_name == 'SL_resnet_untrained_camel_elephant_finetune_vbsle_50k_6way_close_seed77_model_best_4way' :
            checkpoint = torch.load('resnet_untrained_camel_elephant_finetune_vbsle_50k_6way_close_seed77_model_best_4way.pth.tar')
            print(f"loaded resnet_untrained_camel_elephant_finetune_vbsle_50k_6way_close_seed77_model_best_4way") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL_untrained_AppleMesh02' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL_untrained_AppleMesh02.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL_untrained_AppleMesh02") 
        elif model_name == 'SL_resnet50_finetune_vbsl_50k_seed777_model_best_SL_untrained_AppleMesh09' :
            checkpoint = torch.load('resnet50_finetune_vbsl_50k_seed777_model_best_SL_untrained_AppleMesh09.pth.tar')
            print(f"loaded resnet50_finetune_vbsl_50k_seed777_model_best_SL_untrained_AppleMesh09") 
        elif model_name == 'SL_resnet_untrained_camel_elephant_finetune_vbsle_50k_12way_seed77_model_best_12way' :
            checkpoint = torch.load('resnet_untrained_camel_elephant_finetune_vbsle_50k_12way_seed77_model_best_12way_more_epochs.pth.tar')
            print(f"loaded SL_resnet_untrained_camel_elephant_finetune_vbsle_50k_12way_seed77_model_best_12way") 
        elif model_name == 'SL_resnet_untrained_camel_elephant_finetune_vbsle_50k_12way_seed77_model_best_12way' :
            checkpoint = torch.load('resnet_untrained_camel_elephant_finetune_vbsle_50k_12way_seed77_model_best_12way_more_epochs.pth.tar')
            print(f"loaded SL_resnet_untrained_camel_elephant_finetune_vbsle_50k_12way_seed77_model_best_12way") 
        elif model_name == 'SL_untrained_no_texture_texture' :
            checkpoint = torch.load('resnet_untrained_AppleMesh00_AppleMesh01_finetune_vbsle_50k_textured_AppleMesh00_AppleMesh01_seed777_model_best_SL.pth.tar')
            print(f"loaded resnet_untrained_AppleMesh00_AppleMesh01_finetune_vbsle_50k_textured_AppleMesh00_AppleMesh01_seed777_model_best_SL") 
        elif model_name == 'SL_untrained_textured_AppleMesh00_AppleMesh01' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_textured_AppleMesh00_AppleMesh01_seed777_model_best_SL_untrained_textured_AppleMesh02.pth.tar')
            print(f"loaded resnet50_finetune_vbsle_50k_textured_AppleMesh00_AppleMesh01_seed777_model_best_SL_untrained_textured_AppleMesh02") 
        elif model_name == 'SL_untrained_texture_no_texture' :
            checkpoint = torch.load('resnet50_textured_AppleMesh00_AppleMesh01_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL.pth.tar')
            print(f"loaded resnet50_textured_AppleMesh00_AppleMesh01_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel`
        model = model.module        

    elif '12way' in model_name: 
        num_ids = 12
        # init model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_ids)
        model = nn.DataParallel(model).cuda()
        # load model
        if 'SL_resnet50_finetune_vbsle_50k_12way_seed77_model_best_12way' in model_name:
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_12way_seed77_model_best_12way.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_12way_seed77_model_best_12way") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_12way_seed77_model_best_12way_more_epochs' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_12way_seed77_model_best_12way_more_epochs.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_12way_seed77_model_best_12way_more_epochs") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_12way_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_12way_seed77_model_best.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_12way_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_12way_seed77_model_best_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_12way_seed77_model_best_grayscaled.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_12way_seed77_model_best_grayscaled") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel``
        model = model.module
    elif '10way' in model_name: 
        num_ids = 10
        # init model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_ids)
        model = nn.DataParallel(model).cuda()
        # load model
        if model_name == 'SL_resnet50_finetune_vbsle_50k_10way_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_10way_seed77_model_best_10way.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_10way_seed77_model_best") 
        elif model_name == 'SL_barlowtwins_finetune_10way' :
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_10way_seed77_model_best_10way.pth.tar')
            print(f"loaded barlowtwins_finetune_vbsle_50k_10way_seed77_model_best_10way.pth.tar") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_10way_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_10way_seed77_model_best.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_10way_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_10way_seed77_model_best_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_10way_seed77_model_best_grayscaled.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_10way_seed77_model_best_grayscaled") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel``
        model = model.module

    elif '8way' in model_name: 
        num_ids = 8
        # init model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_ids)
        model = nn.DataParallel(model).cuda()
        # load model
        if model_name == 'SL_resnet50_finetune_vbsle_50k_8way_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_8way_seed77_model_best_8way.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_8way_seed77_model_best")
        if model_name == 'SL_resnet50_finetune_vbsle_50k_8way_close_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_8way_close_seed77_model_best_8way.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_8way_seed77_close_model_best") 
        if model_name == 'SL_resnet50_finetune_vbsle_50k_8way_far_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_8way_far_seed77_model_best_8way.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_8way_far_seed77_model_best") 
        elif model_name == 'SL_barlowtwins_finetune_8way' :
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_8way_far_seed77_model_best_8way.pth.tar')
            print(f"loaded barlowtwins_finetune_vbsle_50k_8way_far_seed77_model_best_8way.pth.tar") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_8way_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_8way_seed77_model_best.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_8way_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_8way_seed77_model_best_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_8way_seed77_model_best_grayscaled.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_8way_seed77_model_best_grayscaled") 
       
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_8way_seed77_model_best_blurred_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_8way_seed77_model_best_blurred_grayscaled.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_8way_seed77_model_best_blurred_grayscaled") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_8way_seed77_model_best_blurred' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_8way_seed77_model_best_blurred.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_8way_seed77_model_best_blurred") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_8way_seed77_model_best_blurred_grayscaled_epochs20' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_8way_seed77_model_best_blurred_grayscaled_epochs20.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_8way_seed77_model_best_blurred_grayscaled_epochs20") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel``
        model = model.module
    
    elif '6way' in model_name: 
        num_ids = 6
        # init model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_ids)
        model = nn.DataParallel(model).cuda()
        # load model
        if model_name == 'SL_resnet50_finetune_vbsle_50k_seed77_model_best_6way' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_6way_seed77_model_best_4way.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_seed77_model_best_6way") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_seed77_model_best_6way' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_6way_seed77_model_best_4way.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_seed77_model_best_6way") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_6way_SSDYAJ_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_6way_SSDYAJ_seed77_model_best_4way.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_6way_SSDYAJ_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_6way_close_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_6way_close_seed77_model_best_4way.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_6way_close_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_6way_far_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_6way_far_seed77_model_best_4way.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_6way_far_seed77_model_best") 
        elif model_name == 'SL_barlowtwins_finetune_6way' :
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_6way_far_seed77_model_best.pth.tar')
            print(f"loaded barlowtwins_finetune_vbsle_50k_6way_far_seed77_model_best.pth.tar") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_6way_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_6way_seed77_model_best.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_6way_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_6way_seed77_model_best_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_6way_seed77_model_best_grayscaled.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_6way_seed77_model_best_grayscaled") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_6way_final_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_6way_final_seed77_model_best.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_6way_final_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_6way_final_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_6way_final_colorbg_seed77_model_best.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_6way_final_colorbg_seed77_model_best") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel``
        model = model.module

    elif '4way' in model_name: 
        num_ids = 4
        # init model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_ids)
        model = nn.DataParallel(model).cuda()
        # load model
        if model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_4way' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_4way.pth.tar')
            print(f"loaded resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_4way") 
        if model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_camel_elephant_seed77_model_best_4way' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_camel_elephant_seed77_model_best_4way.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_camel_elephant_seed77_model_best_4way") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_AppleMesh02_AppleMesh06_seed7_model_best_4way' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_AppleMesh02_AppleMesh06_seed7_model_best_4way.pth.tar')
            print(f"loaded resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_AppleMesh02_AppleMesh06_seed7_model_best_4way") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_AppleMesh02_AppleMesh06_seed77_model_best_4way' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_AppleMesh02_AppleMesh06_seed77_model_best_4way.pth.tar')
            print(f"loaded resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_AppleMesh02_AppleMesh06_seed77_model_best_4way") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_AppleMesh02_AppleMesh06_seed777_model_best_4way' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_AppleMesh02_AppleMesh06_seed777_model_best_4way.pth.tar')
            print(f"loaded resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_AppleMesh02_AppleMesh06_seed777_model_best_4way") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_AppleMesh02_AppleMesh05_seed77_model_best_4way' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_AppleMesh02_AppleMesh05_seed77_model_best_4way.pth.tar')
            print(f"loaded resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_AppleMesh02_AppleMesh05_seed77_model_best_4way") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_4way_ADJS_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_4way_ADJS_seed77_model_best_4way.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_4way_ADJS_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_4way_KSSY_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_4way_KSSY_seed77_model_best_4way.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_4way_KSSY_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_4way_KDAS_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_4way_KDAS_seed77_model_best_4way.pth.tar')
            print(f"loaded SL_resnet50_finetune_vbsle_50k_4way_KDAS_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_4way_KDAS_seed77_model_best_onevsall' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_4way_KDAS_seed77_model_best.pth.tar')
            state_dict = checkpoint['state_dict']
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_4way_SSKD_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_4way_SSKD_seed77_model_best.pth.tar')
            state_dict = checkpoint['state_dict']
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_4way_SSKD_colorbg_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_4way_SSKD_colorbg_seed77_model_best.pth.tar')
            state_dict = checkpoint['state_dict']

            # Adjust for DataParallel if needed
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module.backbone."):
                    new_key = k.replace("module.backbone.", "backbone.")
                elif k.startswith("module.classifiers."):
                    new_key = k.replace("module.classifiers.", "classifiers.")
                elif k.startswith("module."):
                    new_key = k.replace("module.", "")
                else:
                    new_key = k
                new_state_dict[new_key] = v
            print("Model loaded successfully.")

        elif model_name == 'SL_barlowtwins_finetune_4way' :
            checkpoint = torch.load('barlowtwins_finetune_vbsle_50k_4way_KDAS_seed77_model_best.pth.tar')
            print(f"loaded barlowtwins_finetune_vbsle_50k_4way_KDAS_seed77_model_best.pth.tar") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_4way_seed77_model_best' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_4way_seed77_model_best.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_4way_seed77_model_best") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_4way_seed77_model_best_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_4way_seed77_model_best_grayscaled.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_4way_seed77_model_best_grayscaled") 
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # roll back from DataParallel``
        model = model.module

    elif '2way' in model_name: 
        num_ids = 2
        # init model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_ids)
        model = nn.DataParallel(model).cuda()
        # load model
        if model_name == 'SL_resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best_layer1' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best_layer1") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best_layer2' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best_layer2") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best_layer3' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best_layer3") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best_layer4_no_pooling' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best_layer4_no_pooling") 
        elif 'SL_resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best_grayscaled' in model_name :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best_grayscaled.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best_grayscaled") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_2way_AppleMesh00_AppleMesh01_seed777_model_best' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_2way_AppleMesh00_AppleMesh01_seed777_model_best.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_2way_AppleMesh00_AppleMesh01_seed777_model_best") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_2way_AppleMesh00_AppleMesh01_seed777_model_best_grayscaled' :
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_2way_AppleMesh00_AppleMesh01_seed777_model_best_grayscaled.pth.tar')
            print(f"loaded SL_resnet50_finetune_texture_colorbg_2way_AppleMesh00_AppleMesh01_seed777_model_best_grayscaled") 
        elif model_name == 'SL_resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best_from_scratch':
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best_from_scratch.pth.tar')
            print("loaded SL_resnet50_finetune_texture_colorbg_2way_AppleMesh02_AppleMesh04_seed777_model_best_from_scratch")
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel``
        model = model.module
  

    elif 'SL_resnet50_finetune_vbsle_50k' in model_name: # 
        model = models.resnet50(pretrained=False)
        if 'trained' in model_name or 'pretrained' in model_name or 'finetune' in model_name:
            model.fc = nn.Linear(2048, 2)
        model = nn.DataParallel(model).cuda()
        if model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best':
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL':
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_seed777_model_best_SL")

        elif model_name == 'SL_resnet50_finetune_vbsle_50k_texture_colorbg_AppleMesh02_AppleMesh10_seed777_model_best':
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_AppleMesh02_AppleMesh10_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_texture_colorbg_AppleMesh02_AppleMesh10_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_texture_colorbg_AppleMesh02_AppleMesh11_seed777_model_best':
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_AppleMesh02_AppleMesh11_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_texture_colorbg_AppleMesh02_AppleMesh11_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_texture_colorbg_AppleMesh02_AppleMesh08_seed777_model_best':
            checkpoint = torch.load('resnet50_finetune_texture_colorbg_AppleMesh02_AppleMesh08_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_texture_colorbg_AppleMesh02_AppleMesh08_seed777_model_best")
        

        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_em_seed777_model_best_SL': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_em_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_em_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh03_em_seed777_model_best_SL': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh03_em_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh03_em_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_half_texture_seed777_model_best_SL': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_half_texture_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh01_half_texture_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_disgust_surprise': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_disgust_surprise_seed777_model_best_SL.pth.tar')
            print("loaded resnet50_finetune_vbsle_50k_disgust_surprise_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh05_seed777_model_best_SL': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh05_seed777_model_best_SL.pth.tar')
            print("loaded resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh05_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh06_seed777_model_best_SL': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh06_seed777_model_best_SL.pth.tar')
            print("loaded resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh06_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh01_AppleMesh06_seed777_model_best_SL': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh01_AppleMesh06_seed777_model_best_SL.pth.tar')
            print("loaded resnet50_finetune_vbsle_50k_AppleMesh01_AppleMesh06_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh01_AppleMesh02_seed777_model_best_SL': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh01_AppleMesh02_seed777_model_best_SL.pth.tar')
            print("loaded resnet50_finetune_vbsle_50k_AppleMesh01_AppleMesh02_seed777_model_best_SL")
        # AppleMesh02 pairs----------------------------------------------------------------------------------------------
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh09_seed777_model_best_SL': # AppleMesh02-AppleMesh09
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh09_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh09_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh07_seed777_model_best_SL': # AppleMesh02-AppleMesh07
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh07_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh07_seed777_model_best_SL")
        elif 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh04_seed777_model_best_SL' in model_name: # AppleMesh02-AppleMesh04
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh04_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh04_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh03_seed777_model_best_SL': # AppleMesh02-AppleMesh03
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh03_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh03_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh05_seed777_model_best_SL': # AppleMesh02-AppleMesh05
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh05_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh05_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh08_seed777_model_best_SL': # AppleMesh02-AppleMesh08
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh08_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh08_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh00_seed777_model_best_SL': # AppleMesh02-AppleMesh00
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh00_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh00_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh10_seed777_model_best_SL': # AppleMesh02-AppleMesh10
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh10_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh10_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh11_seed777_model_best_SL': # AppleMesh02-AppleMesh00
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh11_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh11_seed777_model_best_SL")
        # AppleMesh09 pairs----------------------------------------------------------------------------------------------
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh07_seed777_model_best_SL': # AppleMesh09-AppleMesh07
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh07_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh07_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh04_seed777_model_best_SL': # AppleMesh09-AppleMesh04
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh04_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh04_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh10_seed777_model_best_SL': # AppleMesh09-AppleMesh10
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh10_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh10_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh03_seed777_model_best_SL': # AppleMesh09-AppleMesh03
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh03_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh03_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh05_seed777_model_best_SL': # AppleMesh09-AppleMesh05
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh05_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh05_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh02_seed777_model_best_SL': # AppleMesh09-AppleMesh02
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh02_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh02_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh06_seed777_model_best_SL': # AppleMesh09-AppleMesh06
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh06_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh06_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh08_seed777_model_best_SL': # AppleMesh09-AppleMesh08
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh08_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh08_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh11_seed777_model_best_SL': # AppleMesh09-AppleMesh11
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh11_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh09_AppleMesh11_seed777_model_best_SL")
        # AppleMesh00 pairs ----------------------------------------------------------------------------------------------
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh08_seed777_model_best_SL': # AppleMesh00-AppleMesh08
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh08_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh08_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh03_seed777_model_best_SL': # AppleMesh00-AppleMesh03
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh03_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh03_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh11_seed777_model_best_SL': # AppleMesh00-AppleMesh11
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh11_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh11_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh10_seed777_model_best_SL': # AppleMesh00-AppleMesh10
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh10_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh10_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh07_seed777_model_best_SL': # AppleMesh00-AppleMesh07
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh07_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh07_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh09_seed777_model_best_SL': # AppleMesh00-AppleMesh09
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh09_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh09_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh06_seed777_model_best_SL': # AppleMesh00-AppleMesh06
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh06_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh06_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh00_seed777_model_best_SL': # AppleMesh00-AppleMesh02
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh02_AppleMesh00_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh02_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh04_seed777_model_best_SL': # AppleMesh00-AppleMesh04
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh04_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh00_AppleMesh04_seed777_model_best_SL")
        elif model_name == 'SL_resnet50_finetune_vbsle_50k_textured_AppleMesh00_AppleMesh01_seed777_model_best_SL': # AppleMesh00-AppleMesh04
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_textured_AppleMesh00_AppleMesh01_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_textured_AppleMesh00_AppleMesh01_seed777_model_best_SL")
        
        elif model_name == "SL_resnet50_finetune_vbsle_50k_textured_AppleMesh00_AppleMesh01_seed777_model_best_grayscaled" :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_textured_AppleMesh00_AppleMesh01_seed777_model_best_grayscaled.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_textured_AppleMesh00_AppleMesh01_seed777_model_best_grayscaled")

        elif model_name == "SL_resnet50_finetune_vbsle_50k_AppleMesh03_AppleMesh02_seed777_model_best" :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh03_AppleMesh02_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh03_AppleMesh02_seed777_model_best")
        elif model_name == "SL_resnet50_finetune_vbsle_50k_AppleMesh03_AppleMesh08_seed777_model_best" :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh03_AppleMesh08_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh03_AppleMesh08_seed777_model_best")
        elif model_name == "SL_resnet50_finetune_vbsle_50k_AppleMesh03_AppleMesh08_colorbg_seed777_model_best" :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh03_AppleMesh08_colorbg_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh03_AppleMesh08_colorbg_seed777_model_best")
        elif model_name == "SL_resnet50_finetune_vbsle_50k_AppleMesh03_AppleMesh02_colorbg_seed777_model_best" :
            checkpoint = torch.load('resnet50_finetune_vbsle_50k_AppleMesh03_AppleMesh02_colorbg_seed777_model_best.pth.tar')
            print("loaded SL_resnet50_finetune_vbsle_50k_AppleMesh03_AppleMesh02_colorbg_seed777_model_best")
        
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module
        
    elif 'all' in model_name :
        num_ids = 4
        # init model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_ids)
        model = nn.DataParallel(model).cuda()
        # load model
        checkpoint = torch.load('resnet50_finetune_vbsl_50k_face_obj_all_seed777_model_best.pth.tar')
        print(f"loaded resnet50_finetune_vbsl_50k_face_obj_all_seed777_model_best") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module
    elif '_vs_' in model_name :
        model = models.resnet50(pretrained=False)
        if 'trained' in model_name or 'pretrained' in model_name or 'finetune' in model_name:
            model.fc = nn.Linear(2048, 2)
        model = nn.DataParallel(model).cuda()

        if model_name == 'SL_resnet50_AppleMesh00_vs_AppleMesh01_model_best':
            checkpoint = torch.load('resnet50_AppleMesh00_vs_AppleMesh01_model_best.pth.tar')
            print("loaded resnet50_AppleMesh00_vs_AppleMesh01_model_best")
        elif model_name == 'SL_resnet50_AppleMesh00_vs_camel_model_best':
            checkpoint = torch.load('resnet50_AppleMesh00_vs_camel_model_best.pth.tar')
            print("loaded resnet50_AppleMesh00_vs_camel_model_best")
        elif model_name == 'SL_resnet50_AppleMesh00_vs_elephant_model_best':
            checkpoint = torch.load('resnet50_AppleMesh00_vs_elephant_model_best.pth.tar')
            print("loaded SL_resnet50_AppleMesh00_vs_elephant_model_best")
        elif model_name == 'SL_resnet50_AppleMesh01_vs_camel_model_best':
            checkpoint = torch.load('resnet50_AppleMesh01_vs_camel_model_best.pth.tar')
            print("loaded SL_resnet50_AppleMesh01_vs_camel_model_best")
        elif model_name == 'SL_resnet50_AppleMesh01_vs_elephant_model_best':
            checkpoint = torch.load('resnet50_AppleMesh01_vs_elephant_model_best.pth.tar')
            print("loaded SL_resnet50_AppleMesh01_vs_elephant_model_best")
        elif model_name == 'SL_resnet50_camel_vs_elephant_model_best':
            checkpoint = torch.load('resnet50_camel_vs_elephant_model_best.pth.tar')
            print("loaded SL_resnet50_camel_vs_elephant_model_best")

        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module

    elif 'SL_resnet50_finetune_vbsl_50k' in model_name: # 
        model = models.resnet50(pretrained=False)
        if 'trained' in model_name or 'pretrained' in model_name or 'finetune' in model_name:
            model.fc = nn.Linear(2048, 2)
        model = nn.DataParallel(model).cuda()
        # ----------------------------SL--------------------------------------
        if 'SL_by_step' in model_name :
            num_step = model_name.split("-")[-1]
            checkpoint = torch.load(f'/mnt/smb/locker/issa-locker/users/Seojin/data/saved_models/rn50_preIN_notexture_sizeVar_finetune_vbsl_50k_seed77_step{num_step}_checkpoint.pth.tar')
            print(f"rn50_preIN_notexture_sizeVar_finetune_vbsl_50k_seed77_step{num_step}_checkpoint.pth.tar") 
        if 'SL_by_epoch' in model_name :
            num_epoch = model_name.split("-")[-1]
            checkpoint = torch.load(f'/mnt/smb/locker/issa-locker/users/Seojin/data/saved_models/rn50_preIN_notexture_sizeVar_finetune_vbsl_50k_seed77_epoch{num_epoch}_checkpoint.pth.tar')
            print(f"rn50_preIN_notexture_sizeVar_finetune_vbsl_50k_seed77_epoch{num_epoch}_checkpoint.pth.tar") 
        
        # ----------------------------SL--------------------------------------
        elif model_name == 'SL_resnet50_finetune_vbsl_50k_seed777_model_best':
            checkpoint = torch.load('resnet50_finetune_vbsl_50k_seed777_model_best.pth.tar')
            print("loaded resnet50_finetune_vbsl_50k_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsl_50k_seed77_model_best':
            checkpoint = torch.load('resnet50_finetune_vbsl_50k_seed77_model_best.pth.tar')
            print("loaded resnet50_finetune_vbsl_50k_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsl_50k_obj_seed777_model_best':
            checkpoint = torch.load('resnet50_finetune_vbsl_50k_obj_seed777_model_best.pth.tar')
            print("loaded resnet50_finetune_vbsl_50k_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsl_50k_obj_seed77_model_best':
            checkpoint = torch.load('resnet50_finetune_vbsl_50k_obj_seed77_model_best.pth.tar')
            print("loaded resnet50_finetune_vbsl_50k_seed77_obj_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsl_50k_face_obj_seed77_model_best':
            checkpoint = torch.load('resnet50_finetune_vbsl_50k_face_obj_seed77_model_best.pth.tar')
            print("loaded resnet50_finetune_vbsl_50k_face_obj_seed77_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsl_50k_face_obj_seed777_model_best':
            checkpoint = torch.load('resnet50_finetune_vbsl_50k_face_obj_seed777_model_best.pth.tar')
            print("loaded resnet50_finetune_vbsl_50k_face_obj_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsl_50k_face_obj_seed777_model_best_finetune_vbsl_50k_seed777_model_best':
            checkpoint = torch.load('resnet50_finetune_vbsl_50k_face_obj_seed777_model_best_finetune_vbsl_50k_seed777_model_best.pth.tar')
            print("loaded resnet50_finetune_vbsl_50k_face_obj_seed777_model_best_finetune_vbsl_50k_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsl_50k_face_obj_seed777_model_best_finetune_vbsl_50k_obj_seed777_model_best':
            checkpoint = torch.load('resnet50_finetune_vbsl_50k_face_obj_seed777_model_best_finetune_vbsl_50k_obj_seed777_model_best.pth.tar')
            print("loaded resnet50_finetune_vbsl_50k_face_obj_seed777_model_best_finetune_vbsl_50k_obj_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsl_50k_face_obj_seed777_model_best_finetune_vbsl_50k_obj_seed777_model_best_finetune_vbsl_50k_seed777_model_best':
            checkpoint = torch.load('resnet50_finetune_vbsl_50k_face_obj_seed777_model_best_finetune_vbsl_50k_obj_seed777_model_best_finetune_vbsl_50k_seed777_model_best.pth.tar')
            print("loaded resnet50_finetune_vbsl_50k_face_obj_seed777_model_best_finetune_vbsl_50k_obj_seed777_model_best_finetune_vbsl_50k_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsl_50k_face_obj_seed777_model_best_finetune_vbsl_50k_seed777_model_best_finetune_vbsl_50k_obj_seed777_model_best':
            checkpoint = torch.load('resnet50_finetune_vbsl_50k_face_obj_seed777_model_best_finetune_vbsl_50k_seed777_model_best_finetune_vbsl_50k_obj_seed777_model_best.pth.tar')
            print("loaded resnet50_finetune_vbsl_50k_face_obj_seed777_model_best_finetune_vbsl_50k_seed777_model_best_finetune_vbsl_50k_obj_seed777_model_best")
        
        elif model_name == 'SL_resnet50_finetune_vbsl_50k_seed777_model_best_finetune_vbsl_50k_obj_seed777_model_best': # face->obj
            checkpoint = torch.load('resnet50_finetune_vbsl_50k_seed777_model_best_finetune_vbsl_50k_obj_seed777_model_best.pth.tar')
            print("loaded resnet50_finetune_vbsl_50k_seed777_model_best_finetune_vbsl_50k_obj_seed777_model_best")
        elif model_name == 'SL_resnet50_finetune_vbsl_50k_obj_seed777_model_best_finetune_vbsl_50k_seed777_model_best': # obj->face
            checkpoint = torch.load('resnet50_finetune_vbsl_50k_obj_seed777_model_best_finetune_vbsl_50k_seed777_model_best.pth.tar')
            print("loaded resnet50_finetune_vbsl_50k_obj_seed777_model_best_finetune_vbsl_50k_seed777_model_best")

        elif model_name == 'SL_resnet50_finetune_vbsl_50k_obj_seed777_model_best_finetune_vbsl_50k_seed777_model_best': # obj->face
            checkpoint = torch.load('resnet50_finetune_vbsl_50k_obj_seed777_model_best_finetune_vbsl_50k_seed777_model_best.pth.tar')
            print("loaded resnet50_finetune_vbsl_50k_obj_seed777_model_best_finetune_vbsl_50k_seed777_model_best")
    
        elif model_name == 'SL_resnet50_finetune_vbsl_50k_seed777_model_best_SL': # retrained 0927
            checkpoint = torch.load('resnet50_finetune_vbsl_50k_seed777_model_best_SL.pth.tar')
            print("loaded SL_resnet50_finetune_vbsl_50k_seed777_model_best_SL")

       

        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module
    elif "multitask" in model_name :
        # SL --------

        model = models.resnet50(pretrained=False)
        if 'trained' in model_name or 'pretrained' in model_name or 'finetune' in model_name:
            model.fc = nn.Linear(2048, 2)
        model = nn.DataParallel(model).cuda()
        checkpoint = torch.load('resnet50_finetune_model_best_multitask50k.pth.tar')
        model.load_state_dict(checkpoint['state_dict'],  strict=False)
        # roll back from DataParallel
        model = model.module
        print("loaded resnet50_finetune_model_best_multitask50k")

        # -----------



        # checkpoint = torch.load('resnet50_finetune_model_best_multitask50k.pth.tar')
        # print("loaded resnet50_finetune_model_best_multitask50k")

        # # Load state dict
        # state_dict = checkpoint['state_dict']
        
        # # Create a new state dict with the 'module.' prefix added to 'shared_layers' keys
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.startswith('module.shared_layers.'):
        #         new_state_dict['module.' + k] = v
        #     if k.startswith('shared_layers.'):
        #         new_key = 'module.' + k[len('shared_layers.'):]
        #         new_state_dict[new_key] = v
        #     if k.startswith('task_a_fc.'):
        #         new_key = 'module.fc.' + k[len('task_a_fc.'):]
        #         new_state_dict[new_key] = v
        #     else:
        #         new_state_dict[k] = v
        #         print(k)
        # model = models.resnet50(pretrained=False)

        # model = nn.DataParallel(model).cuda()
        # model.load_state_dict(new_state_dict)
        # model = model.module

>>>>>>> 535f7fc (temp update)
    # finetuning steps

    elif 'step' in model_name:
        # init model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 2)
        model = nn.DataParallel(model).cuda()
        # load model
        dataset = model_name.split("_")[0]
        if dataset == 'vbsl50k':
            dataset = 'vbsl_50k'
        elif dataset == 'vbsl50kobj':
            dataset = 'vbsl_50k_obj'
        else:
            assert False
        step = model_name.split("_")[-1][4:]
        filename = f'./saved_models/finetuning_steps/resnet50_finetune_{dataset}_seed7_step{step}_checkpoint.pth.tar'
        checkpoint = torch.load(filename)
        print(f"loaded {filename}, epoch{checkpoint['epoch']}") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module
    
    # num ids ablation
    elif "id" in model_name: # vbsl5k_basel_id{num_ids}_seed7
        num_ids = int(model_name.split("_")[2][2:])
        seed = int(model_name.split("_")[3][4:])
        # init model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_ids)
        model = nn.DataParallel(model).cuda()
        # load model
        epoch = int(120/num_ids)
        filename = f'./saved_models/num_ids_ablation/resnet50_finetune_vbsl5k_id{num_ids}_epoch{epoch}_seed{seed}_model_best.pth.tar'
        checkpoint = torch.load(filename)
        print(f"loaded {filename}, epoch{checkpoint['epoch']}") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module
    
    elif model_name == 'Basel_50k_2id':
        # init model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 2)
        model = nn.DataParallel(model).cuda()
        # load model
        num_epoch = model_name.split("_")[-1]
        filename = f'./saved_models/resnet50_finetune_vbsl_50k_Basel01_epoch24_model_best.pth.tar'
        checkpoint = torch.load(filename)
        print(f"loaded {filename}") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module
<<<<<<< HEAD
=======

    elif "Basel_color_texture" in model_name :
        model = models.resnet50(pretrained=False)
        if '10id' in model_name :
            model.fc = nn.Linear(2048, 10)
        elif '20ID' in model_name :
            model.fc = nn.Linear(2048,20)
        elif '30ID' in model_name :
            model.fc = nn.Linear(2048,30)
        elif '40ID' in model_name :
            model.fc = nn.Linear(2048,40)
        elif '50ID' in model_name :
            model.fc = nn.Linear(2048,50)
        elif '6id' in model_name :
            model.fc = nn.Linear(2048, 6)
        elif '100id' in model_name :
            model.fc = nn.Linear(2048, 100)
        elif '500id' in model_name :
            model.fc = nn.Linear(2048, 500)
        elif '1kid' in model_name :
            model.fc = nn.Linear(2048, 1000)
        else :
            model.fc = nn.Linear(2048, 2)
        model = nn.DataParallel(model).cuda()
        if model_name == "SL_resnet50_finetune_Basel_color_texture_1kid_seed777_model_best" :
            checkpoint = torch.load("original_resnet50_finetune_Basel_color_texture_1kid_seed777_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_color_texture_1kid_seed777__from_scratch" :
            checkpoint = torch.load("resnet50_finetune_Basel_color_texture_1kid_seed777__from_scratch.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_color_texture_100id_seed777_model_best" :
            checkpoint = torch.load("resnet50_finetune_Basel_color_texture_100id_seed777_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_color_texture_500id_seed777__best" :
            checkpoint = torch.load("resnet50_finetune_Basel_color_texture_500id_seed777__best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_color_texture_2id_seed777_model_best" :
            checkpoint = torch.load("resnet50_finetune_Basel_color_texture_2id_seed777_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_color_texture_10id_seed77_model_best" :
            checkpoint = torch.load("resnet50_finetune_Basel_color_texture_10id_seed77_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_color_texture_10id_seed77_model_best" :
            checkpoint = torch.load("resnet50_finetune_Basel_color_texture_10id_seed77_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_color_texture_6id_seed77_model_best" :
            checkpoint = torch.load("resnet50_finetune_Basel_color_texture_6id_seed77_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_color_texture_30ID_seed777" :
            checkpoint = torch.load("resnet50_finetune_Basel_color_texture_30ID_seed777_.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_color_texture_40ID_seed777" :
            checkpoint = torch.load("resnet50_finetune_Basel_color_texture_40ID_seed777_.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_color_texture_50ID_seed777" :
            checkpoint = torch.load("resnet50_finetune_Basel_color_texture_50ID_seed777_.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_color_texture_20ID_seed777" :
            checkpoint = torch.load("resnet50_finetune_Basel_color_texture_20ID_seed777_.pth.tar")
            print(f"loaded {model_name}") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module

    elif "Basel_texture" in model_name :
        model = models.resnet50(pretrained=False)
        if '4ID' in model_name :
            model.fc = nn.Linear(2048, 4)
        elif '6ID' in model_name :
            model.fc = nn.Linear(2048, 6)
        elif '8ID' in model_name :
            model.fc = nn.Linear(2048, 8)
        elif '10ID' in model_name :
            model.fc = nn.Linear(2048, 10)
        elif '12ID' in model_name :
            model.fc = nn.Linear(2048, 12)
        elif '20ID' in model_name :
            model.fc = nn.Linear(2048, 20)
        elif '100ID' in model_name :
            model.fc = nn.Linear(2048, 100)
        elif '500ID' in model_name :
            model.fc = nn.Linear(2048, 500)
        elif '1000ID' in model_name :
            model.fc = nn.Linear(2048, 999)
        else :
            model.fc = nn.Linear(2048, 2)
        model = nn.DataParallel(model).cuda()
        if model_name == "SL_resnet50_finetune_Basel_texture_2ID_seed777_model_best_SL" :
            checkpoint = torch.load("resnet50_finetune_Basel_texture_2ID_seed777_model_best_SL.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_texture_4ID_seed777_model_best_SL" :
            checkpoint = torch.load("resnet50_finetune_Basel_texture_4ID_seed77_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_texture_6ID_seed777_model_best_SL" :
            checkpoint = torch.load("resnet50_finetune_Basel_texture_6ID_seed77_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_texture_8ID_seed777_model_best_SL" :
            checkpoint = torch.load("resnet50_finetune_Basel_texture_8ID_seed77_model_best_8way.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_texture_10ID_seed777_model_best_SL" :
            checkpoint = torch.load("resnet50_finetune_Basel_texture_10ID_seed77_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_texture_12ID_seed777_model_best_SL" :
            checkpoint = torch.load("resnet50_finetune_Basel_texture_12ID_seed77_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == 'SL_resnet50_finetune_Basel_texture_20ID_seed77_model_best' :
            checkpoint = torch.load("resnet50_finetune_Basel_texture_20ID_seed77_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == 'SL_resnet50_finetune_Basel_texture_20ID_seed777_model_best' :
            checkpoint = torch.load("resnet50_finetune_Basel_texture_20ID_seed777_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == 'SL_resnet50_finetune_Basel_texture_100ID_seed77_model_best' :
            checkpoint = torch.load("resnet50_finetune_Basel_texture_100ID_seed77_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == 'SL_resnet50_finetune_Basel_texture_100ID_seed777_model_best' :
            checkpoint = torch.load("resnet50_finetune_Basel_texture_100ID_seed777_model_best.pth.tar")
            print(f"loaded {model_name}")      
        elif model_name == 'SL_resnet50_finetune_Basel_texture_1000ID_seed777_model_best' :
            checkpoint = torch.load("resnet50_finetune_Basel_texture_1000ID_seed777_model_best_grayscaled_epoch50.pth.tar") # wrong name, it was 2 epochs not grayscaled
            print(f"loaded {model_name}")    
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module

    elif "Basel_no_texture" in model_name :
        model = models.resnet50(pretrained=False)
        if '4ID' in model_name :
            model.fc = nn.Linear(2048, 4)
        elif '6ID' in model_name :
            model.fc = nn.Linear(2048, 6)
        elif '8ID' in model_name :
            model.fc = nn.Linear(2048, 8)
        elif '10ID' in model_name :
            model.fc = nn.Linear(2048, 10)
        elif '12ID' in model_name :
            model.fc = nn.Linear(2048, 12)
        elif '20ID' in model_name :
            model.fc = nn.Linear(2048, 20)
        elif '30ID' in model_name :
            model.fc = nn.Linear(2048, 30)
        elif '40ID' in model_name :
            model.fc = nn.Linear(2048, 40)
        elif '50ID' in model_name :
            model.fc = nn.Linear(2048, 50)
        elif '100ID' in model_name :
            model.fc = nn.Linear(2048, 100)
        elif '500ID' in model_name :
            model.fc = nn.Linear(2048, 500)
        elif '1000ID' in model_name :
            model.fc = nn.Linear(2048, 1000)
        else :
            model.fc = nn.Linear(2048, 2)
        model = nn.DataParallel(model).cuda()
        if model_name == "SL_resnet50_finetune_Basel_no_texture_2ID_seed777_model_best_SL" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_2ID_seed777_model_best_SL.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_4ID_seed777_model_best_SL" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_4ID_seed77_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_6ID_seed777_model_best_SL" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_6ID_seed77_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_8ID_seed777_model_best_SL" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_8ID_seed77_model_best_8way.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_10ID_seed777_model_best_SL" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_10ID_seed77_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_12ID_seed77_model_best" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_12ID_seed77_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_20ID_seed77_model_best" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_20ID_seed77_model_best.pth.tar")
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_20ID_seed777_model_best" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_20ID_seed777_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_100ID2_seed77_model_best_epochs_10" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_100ID2_seed77_model_best_epochs_10.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_100ID2_seed77_model_best" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_100ID2_seed77_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_100ID2_seed777_model_best" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_100ID2_seed777_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_500ID2_seed77_model_best" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_500ID2_seed77_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_500ID2_seed77_model_best2" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_500ID2_seed77_model_best2.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_500ID2_seed77_model_best_epochs_10" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_500ID2_seed77_model_best_epochs_10.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_1000ID_smaller_seed777_model_best" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_1000ID_smaller_seed777_model_best.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_1000ID_seed777_model_best" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_1000ID_seed777_model_best_grayscaled_epoch50.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_30ID_seed777_" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_30ID_seed777_.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_40ID_seed777_" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_40ID_seed777_.pth.tar")
            print(f"loaded {model_name}") 
        elif model_name == "SL_resnet50_finetune_Basel_no_texture_50ID_seed777_" :
            checkpoint = torch.load("resnet50_finetune_Basel_no_texture_50ID_seed777_.pth.tar")
            print(f"loaded {model_name}") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module
        
    # num ids ablation
    elif "id" in model_name: # vbsl5k_basel_id{num_ids}_seed7
        num_ids = int(model_name.split("_")[2][2:])
        seed = int(model_name.split("_")[3][4:])
        # init model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_ids)
        model = nn.DataParallel(model).cuda()
        # load model
        epoch = int(120/num_ids)
        filename = f'/mnt/smb/locker/issa-locker/users/AppleMesh09/data/saved_models/num_ids_ablation/resnet50_finetune_vbsl5k_id{num_ids}_epoch{epoch}_seed{seed}_model_best.pth.tar'
        checkpoint = torch.load(filename)
        print(f"loaded {filename}, epoch{checkpoint['epoch']}") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module

>>>>>>> 535f7fc (temp update)
    
    # subset ablation
    elif 'vbsl50k_subset' in model_name: # vbsl50k_subset_0.05
        # init model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 2)
        model = nn.DataParallel(model).cuda()
        # load model
        ratio = model_name.split("_")[-1]
        epoch = int(6/float(ratio))
        filename = f'./saved_models/subset_ablation/resnet50_finetune_vbsl_50k_epoch{epoch}_seed7_subset{ratio}_model_best.pth.tar'
        checkpoint = torch.load(filename)
        print(f"loaded {filename}, epoch{checkpoint['epoch']}") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module
    
    # finetuning epochs
    elif 'PennTuning' in model_name:
        # init model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 2)
        model = nn.DataParallel(model).cuda()
        # load model
        num_epoch = model_name.split("-")[-1]
        if 'face' in model_name:
            filename = f'./saved_models/finetuning_penn/resnet50_finetune_vbsl_50k_epoch24_{num_epoch}_checkpoint.pth.tar'
        elif 'obj' in model_name:
            filename = f'./saved_models/finetuning_penn/resnet50_finetune_vbsl_50k_obj_epoch24_{num_epoch}_checkpoint.pth.tar'
        checkpoint = torch.load(filename)
        print(f"loaded {filename}") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module
    
    # vbsl-dist finetuning epochs
    elif 'rn50_vbsl-dist-ft' in model_name:
        # init model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 2)
        model = nn.DataParallel(model).cuda()
        # load model
        num_epoch = model_name.split("_")[-1]
        filename = f'./saved_models/resnet50_finetune_vbsl_dist_50k_{num_epoch}_checkpoint.pth.tar'
        checkpoint = torch.load(filename)
        print(f"loaded {filename}") 
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module
    elif model_name in ['resnet50barlowtwins']:
        model = models.__dict__['resnet50'](pretrained=False)
        ckpt = torch.load('./saved_models/resnet50-barlowtwins.pth')
        model.load_state_dict(ckpt, strict=False)
        model.fc = nn.Linear(2048, 2)
        model = torch.nn.DataParallel(model).cuda()
        model = model.module
        print(f"loaded barlowtwins resnet50")
    # SIN pretrained models
    elif model_name in ['resnet50-SIN', 'resnet50-SIN-IN', 'resnet50-SIN-IN-ft']:
        model = models.resnet50(pretrained=False)
        model = torch.nn.DataParallel(model).cuda()
        if model_name == 'resnet50-SIN':
            URL = model_urls['resnet50_trained_on_SIN']
        elif model_name == 'resnet50-SIN-IN':
            URL = model_urls['resnet50_trained_on_SIN_and_IN']
        elif model_name == 'resnet50-SIN-IN-ft':
            URL = model_urls['resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN']
        else:
            assert(False)
        checkpoint = model_zoo.load_url(URL)
        model.load_state_dict(checkpoint["state_dict"])
        # roll back from DataParallel
        model = model.module
    
    # dm_pred tuning
    elif 'dm_pred_tuning' in model_name:
        model = rn50_auxiliary_dm()
        model.fc = nn.Linear(2048, 1000)
        model = torch.nn.DataParallel(model).cuda()
        # load ckpt
        tuning_para = model_name.split("_")[-1]
        # checkpoint = torch.load(f'./saved_models/dm_pred_tuning/TUNING_{tuning_para}_rn50DM_notexture_sizeVar_stage2_2losses_model_best.pth.tar')
        checkpoint = torch.load(f'./saved_models/dm_pred_tuning/TUNING_{tuning_para}_rn50DM_notexture_sizeVar_stage2_2losses_checkpoint.pth.tar')
        print("loaded model_name")
        
        model.load_state_dict(checkpoint["state_dict"])
        model = model.module
    
    # pretrained on basal faces - with DM
    elif model_name in ['rn50_preIN_notexture_sizeVar_2losses',
                        'rn50_preIN_notexture_sizeVar_onlyDM']:
        model = rn50_auxiliary_dm()
        model.fc = nn.Linear(2048, 1000)
        model = torch.nn.DataParallel(model).cuda()
        
        if model_name == 'rn50_preIN_notexture_sizeVar_2losses':
            checkpoint = torch.load('./saved_models/rn50DM_notexture_sizeVar_stage2_2losses_model_best.pth.tar')
            print("loaded rn50_preIN_notexture_sizeVar_2losses")
        elif model_name == 'rn50_preIN_notexture_sizeVar_onlyDM':
            checkpoint = torch.load('./saved_models/rn50DM_notexture_sizeVar_stage2_onlyDM_checkpoint.pth.tar')
            print("loaded rn50_preIN_notexture_sizeVar_onlyDM")
        else:
            assert False
        model.load_state_dict(checkpoint["state_dict"])
        model = model.module
        
    
    elif 'simplecnn' in model_name:
        from models.simplecnn import SimpleCNN
        model = SimpleCNN()
        model = torch.nn.DataParallel(model).cuda()
        # get checkpoint
        if model_name == 'simplecnn-trained-vbsl':
            checkpoint = torch.load('./saved_models/simplecnn_256bs_from_scratch_model_best.pth.tar')
        else:
            assert False
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module
        
    elif 'trained' in model_name or 'rn50' in model_name:
        model = models.resnet50(pretrained=False)
        if 'trained' in model_name or 'pretrained' in model_name or 'finetune' in model_name:
            model.fc = nn.Linear(2048, 2)
        model = nn.DataParallel(model).cuda()
        # vbsl trained 
        if 'resnet50-trained-pretrained-vbsl' in model_name:
            checkpoint = torch.load('./saved_models/256bs_pretrained_model_best.pth.tar')
            print("loaded 256bs_pretrained_model_best.pth.tar")
        elif model_name == 'resnet50-trained-scratch-vbsl':
            checkpoint = torch.load('./saved_models/256bs_from_scratch_model_best.pth.tar')
            print("loaded 256bs_from_scratch_model_best.pth.tar")
        # NEW obj vbsl trained - finetuning epochs
        elif 'resnet50-trained-pretrained-vbsl_obj' in model_name:
            num_epoch = model_name.split("-")[-1]
            checkpoint = torch.load(f'./saved_models/resnet50_pretrained_vbsl_50k_obj_epoch{num_epoch}_checkpoint.pth.tar')
            print(f"resnet50_pretrained_vbsl_50k_obj_epoch{num_epoch}_checkpoint.pth.tar") 
        elif "resnet50barlowtwins-trained-pretrained-vbsl_obj" in model_name:
            num_epoch = model_name.split("-")[-1]
            checkpoint = torch.load(f'./saved_models/resnet50barlowtwins_pretrained_vbsl_50k_obj_epoch{num_epoch}_checkpoint.pth.tar')
            print(f"resnet50barlowtwins_pretrained_vbsl_50k_obj_epoch{num_epoch}_checkpoint.pth.tar") 
        # NEW vbsl trained - finetuning epochs
        elif 'resnet50-trained-pretrained-vbsl' in model_name:
            num_epoch = model_name.split("-")[-1]
            checkpoint = torch.load(f'./saved_models/resnet50_pretrained_vbsl_50k_epoch{num_epoch}_checkpoint.pth.tar')
            print(f"resnet50_pretrained_vbsl_50k_epoch{num_epoch}_checkpoint.pth.tar") 
        elif 'resnet50barlowtwins_pretrained_vbsl' in model_name:
            num_epoch = model_name.split("-")[-1]
            checkpoint = torch.load(f'./saved_models/resnet50barlowtwins_pretrained_vbsl_50k_epoch{num_epoch}_checkpoint.pth.tar')
            print(f"resnet50barlowtwins_pretrained_vbsl_50k_epoch{num_epoch}_checkpoint.pth.tar") 
        elif 'rn50_preIN_notexture_sizeVar_finetune_vbsl_50k' in model_name: # 
            num_epoch = model_name.split("-")[-1]
            checkpoint = torch.load(f'./saved_models/finetuning_tuning/rn50_preIN_notexture_sizeVar_finetune_vbsl_50k_epoch{num_epoch}_checkpoint.pth.tar')
            print(f"rn50_preIN_notexture_sizeVar_finetune_vbsl_50k_epoch{num_epoch}_checkpoint.pth.tar") 
        # basal pretrained
        # elif model_name == 'rn50_preIN_notexture_model_best':
        #     checkpoint = torch.load('./saved_models/rn50_preIN_notexture_model_best.pth.tar')
        #     print("loaded rn50_preIN_notexture_model_best.pth.tar")
        # elif model_name == 'rn50_preIN_texture_model_best':
        #     checkpoint = torch.load('./saved_models/rn50_preIN_texture_model_best.pth.tar')
        #     print("loaded rn50_preIN_texture_model_best.pth.tar")
        elif model_name == 'rn50_preIN_notexture_sizeVar_best':
            checkpoint = torch.load('./saved_models/rn50_preIN_notexture_sizeVar_model_best.pth.tar')
            print("loaded rn50_preIN_notexture_sizeVar_best")
        elif model_name == 'rn50_preIN_texture_sizeVar_best':
            checkpoint = torch.load('./saved_models/rn50_preIN_texture_sizeVar_model_best.pth.tar')
            print("loaded rn50_preIN_notexture_sizeVar_best")
        elif model_name == 'rn50_FromScratch_notexture_sizeVar_best':
            checkpoint = torch.load('./saved_models/rn50_FromScratch_notexture_sizeVar_model_best.pth.tar')
            print("loaded rn50_FromScratch_notexture_sizeVar_best")
        elif model_name == 'rn50_FromScratch_texture_sizeVar_best':
            checkpoint = torch.load('./saved_models/rn50_FromScratch_texture_sizeVar_model_best.pth.tar')
            print("loaded rn50_FromScratch_texture_sizeVar_best")
        else:
            assert(False)
        model.load_state_dict(checkpoint['state_dict'])
        # roll back from DataParallel
        model = model.module
    else:
        assert False, f"unsupported model type: {model_name}"
    model.eval()
    return model

class Vgg_face_dag(nn.Module):

    def __init__(self):
        super(Vgg_face_dag, self).__init__()
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward(self, x0):
        x1 = self.conv1_1(x0)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)
        x31_preflatten = self.pool5(x30)
        x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
        x32 = self.fc6(x31)
        x33 = self.relu6(x32)
        x34 = self.dropout6(x33)
        x35 = self.fc7(x34)
        x36 = self.relu7(x35)
        x37 = self.dropout7(x36)
        x38 = self.fc8(x37)
        return x38
