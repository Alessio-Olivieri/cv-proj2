# %%
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts
sys.path.append('../src')
from modules import (
                    paths,
                    dataset,
                    model,
                    utils,
                    acdc,
                    train
                    )
from torchvision.transforms import v2
from torch.optim import AdamW

# %%
transform_train = v2.Compose([
    v2.Lambda(lambda x: x.convert('RGB')),  # some images are in grayscale
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(),
    v2.RandAugment(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.RandomErasing(p=0.25),

])

transform_valid = v2.Compose([
    v2.Lambda(lambda x: x.convert('RGB')),  # some images are in grayscale
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# %%
toy=False

# %%
import importlib
importlib.reload(dataset)
if toy == True:
    print("laoding toy datasets")
    train_dataset, coarse_labels = dataset.load_animal_dataset("train", transform=transform_train, tiny=True, stop=6)
    val_dataset, coarse_labels = dataset.load_animal_dataset("valid", transform=transform_valid, tiny=True, stop=2)

else:
    print("loading full dataet")
    train_dataset, coarse_labels = dataset.load_animal_dataset("train", transform=transform_train)
    val_dataset, coarse_labels = dataset.load_animal_dataset("valid", transform=transform_valid)

train_dataset = dataset.TorchDatasetWrapper(train_dataset, transform=transform_train)
val_dataset = dataset.TorchDatasetWrapper(val_dataset, transform=transform_valid)
print("train:\n"+str(train_dataset))
print("validation:\n"+str(val_dataset))


# %%
batch_size = 5 if toy else 4096 

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,  
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,  
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True
)

# %% [markdown]
# Train the model

# %%
importlib.reload(model)
config = {
    "patch_size": 8,           # Kept small for fine-grained patches
    "hidden_size": 64,          # Increased from 48 (better representation)
    "num_hidden_layers": 6,     # Deeper for pruning flexibility
    "num_attention_heads": 8,   # More heads (head_dim = 64/8 = 8)
    "intermediate_size": 4 * 64,# Standard FFN scaling
    "hidden_dropout_prob": 0.2, # Mild dropout for regularization
    "attention_probs_dropout_prob": 0.2,
    "initializer_range": 0.02,
    "image_size": 64,
    "num_classes": 58,
    "num_channels": 3,
    "qkv_bias": True,           # Keep bias for now (can prune later)
}

importlib.reload(train)

class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy loss compatible with Mixup/Cutmix soft labels"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x, target):
        # x = model outputs (logits)
        # target = mixed labels (probability distributions)
        loss = torch.sum(-target * F.log_softmax(x, dim=1), dim=1)
        return loss.mean()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
vit = model.ViT(config).to(device)

num_epochs = 500
warmup_epochs = 20
base_lr = 3e-4
min_lr = 1e-6
weight_decay = 0.05  # For AdamW optimizer
label_smoothing = 0.1  # For cross-entropy
patience = 50



optimizer = AdamW(vit.parameters(),
                  lr=base_lr,
                  weight_decay = weight_decay,
                  betas=(0.9, 0.98),
                  eps = 1e-6      
                  )

# Linear warmup for 30 epochs (0 → base_lr)
warmup = LinearLR(
    optimizer,
    start_factor=1e-6,  # Near-zero initial LR
    end_factor=1.0,     # Full LR after warmup
    total_iters=warmup_epochs,
)

# Cosine decay for remaining epochs (170)
cosine = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=num_epochs - warmup_epochs,  # 170 epochs per cycle
    eta_min=min_lr,
)

# Combine them
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[warmup_epochs],  # Switch after warmup
)

mixup_fn = v2.MixUp(
    alpha=1.0,          # Add CutMix
    num_classes=58
)

trainer = train.Trainer(model=vit,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        criterion=SoftTargetCrossEntropy(),
                        val_criterion=nn.CrossEntropyLoss(),
                        scheduler=scheduler,
                        device = device,
                        writer=torch.utils.tensorboard.SummaryWriter(log_dir=paths.logs),
                        scaler=torch.amp.GradScaler(),
                        num_epochs=num_epochs,
                        log_interval=50,
                        model_dir=paths.chekpoints,
                        mixup_fn=mixup_fn,
                        early_stop_patience=patience,
                        model_name="vit1.pth",
                        resume=False
                        )


# %%
# acc = trainer.train()

# %%
importlib.reload(dataset)
train_subset, coarse_labels = dataset.load_animal_dataset("train", transform=transform_train, tiny=True, stop=50)# as big as validation
print(coarse_labels)

# %%
importlib.reload(acdc)
importlib.reload(dataset)
train_subset, coarse_labels = dataset.load_animal_dataset("train", transform=transform_train, tiny=True, stop=50)# as big as validation
train_subset = dataset.TorchDatasetWrapper(train_subset, transform=transform_valid)

acdc_dataset = dataset.ContrastiveWrapper(train_subset, coarse_labels)

acdc_loader = DataLoader(
    acdc_dataset,
    batch_size=200,
    shuffle=True,
    num_workers=4,  
    pin_memory=True,
    prefetch_factor=4,
    collate_fn=dataset.contrastive_collate_fn,
)
clean_batch, corrupted_batch = next(iter(acdc_loader))


# %%
trainer.load_checkpoint(paths.chekpoints/"vit1.pth")
vit = trainer.model

# %%
importlib.reload(acdc)
importlib.reload(utils)

circuits = {}
for tau in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    circuits[str(tau)] = acdc.run_ACDC_optimized(vit, tau, acdc_loader, device=device)

import pickle
print("saving in circuits.pkl")
pickle.dump(circuits, open("circuits.pkl", "wb"))
print("saved in circuits.pkl")



