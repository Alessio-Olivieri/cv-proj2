import vit_prisma
from vit_prisma.utils import prisma_utils

import numpy as np
import torch
from fancy_einsum import einsum
from collections import defaultdict

import plotly.graph_objs as go
import plotly.express as px

import matplotlib.colors as mcolors

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from IPython.core.display import display, HTML

import timm

def patch_model(model, prisma=False, timm=False):
    assert (prisma or timm)
    if prisma:
        model.pos_embed.W_pos = torch.nn.Parameter(torch.zeros(65, 768))
        model.embed.proj = torch.nn.Conv2d(3, 768, kernel_size=(8, 8), stride=(8, 8))
        model.head.W_H = torch.nn.Parameter(torch.empty(768, 200))
        model.head.b_H = torch.nn.Parameter(torch.zeros(200))
    elif timm:
        model.pos_embed = torch.nn.Parameter(torch.zeros(1, 65, 768))
        model.patch_embed.proj = torch.nn.Conv2d(3, 768, kernel_size=(8, 8), stride=(8, 8))
    return model

from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from transformers import AutoConfig
def convert_pretrained_model_config(model, model_name="vit_base_patch16_384", is_timm: bool = True, is_clip: bool = False) -> HookedViTConfig:
    hf_config = AutoConfig.from_pretrained(model.default_cfg['hf_hub_id'])

    if hasattr(hf_config, 'patch_size'):
        ps = hf_config.patch_size
    elif hasattr(hf_config, "tubelet_size"):
        ps = hf_config.tubelet_size[1]

    pretrained_config = {
        'n_layers' : hf_config.num_hidden_layers if not is_timm else len(model.blocks),
        'd_model' : hf_config.hidden_size if not is_timm else model.embed_dim,
        'd_head' : hf_config.hidden_size // hf_config.num_attention_heads if not is_timm else model.embed_dim // model.blocks[0].attn.num_heads,
        'model_name' : hf_config._name_or_path,
        'n_heads' : hf_config.num_attention_heads if not is_timm else model.blocks[0].attn.num_heads,
        'd_mlp' : hf_config.intermediate_size if not is_timm else 4 * model.embed_dim,
        'activation_name' : hf_config.hidden_act if not is_timm else model.blocks[0].mlp.act.__class__.__name__.lower(),
        'eps' : getattr(hf_config, 'layer_norm_eps', 1e-12),
        'original_architecture' : getattr(hf_config, 'architecture', getattr(hf_config, 'architectures', None)),
        'initializer_range' : hf_config.initializer_range,
        'n_channels' : hf_config.num_channels if not is_timm else model.patch_embed.proj.weight.shape[1],
        'patch_size' : ps if not is_timm else model.patch_embed.proj.kernel_size[0],
        'image_size' : hf_config.image_size if not is_timm else model.default_cfg['input_size'][1],
        'n_classes' :  getattr(hf_config, "num_classes", None) if not is_timm else model.default_cfg['num_classes'],
        'n_params' : sum(p.numel() for p in model.parameters() if p.requires_grad) if is_timm else None,
    }
    
    # Rectifying Huggingface bugs:
    # Currently a bug getting configs, only this model confirmed to work and even it requires modification of eps
    if is_timm and model_name == "vit_base_patch16_224":
        pretrained_config.update({
            "eps": 1e-6,
            "return_type": "class_logits",
        })
    
    # Config for 32 is incorrect, fix manually 
    if is_timm and model_name == "vit_base_patch32_224":
        pretrained_config.update({
            "patch_size": 32,
            "eps": 1e-6,
            "return_type": "class_logits"
        })

    if is_clip:
        pretrained_config.update({
            "layer_norm_pre": True,
            "return_type": "class_logits" # actually returns 'visual_projection'
        })

    if "dino" in model_name:
        pretrained_config.update({
            "return_type": "pre_logits",
            "n_classes": 768,
        })

    # Config is for ViVet, need to add more properties
    if hasattr(hf_config, "tubelet_size"):
        pretrained_config.update({
            "is_video_transformer": True,
            "video_tubelet_depth": hf_config.tubelet_size[0],
            "video_num_frames": hf_config.video_size[0],
            "n_classes": 400 if "kinetics400" in model_name else None,
            "return_type": "class_logits" if "kinetics400" in model_name else "pre_logits",

        })

    if pretrained_config['n_classes'] is None:
        id2label = getattr(hf_config, "id2label", None)
        if id2label is not None:
            pretrained_config.update({
                "n_classes": len(id2label),
                "return_type": "class_logits"
            })
    
    return HookedViTConfig.from_dict(pretrained_config)


torch.backends.cudnn.benchmark = True

NUM_CLASSES = 200
BATCH_SIZE = 64  
NUM_EPOCHS = 10  
LEARNING_RATE = 1e-4  
WEIGHT_DECAY = 0.01  
IMAGE_SIZE = 384
LOG_INTERVAL = 100 
DATA_DIR = "data/"
MODEL_DIRS = DATA_DIR+"models/"
BASE_MODEL = MODEL_DIRS+"interpolated_vit_tiny_imagenet.pth"
TRAIN_DATA = DATA_DIR+"train.pkl"
VALID_DATA = DATA_DIR+"valid.pkl"

def get_model(num_classes=200):
    model = patch_model(HookedViT.from_pretrained("vit_base_patch16_384",
                                            center_writing_weights=True,
                                            center_unembed=True,
                                            fold_ln=True,
                                            refactor_factored_attn_matrices=True,
                                            num_classes=NUM_CLASSES
                                              
                                        ),
                       prisma=True
                       )
    return model
    
