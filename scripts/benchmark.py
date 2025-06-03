#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset
import PIL
import torch
from torchvision.transforms import v2
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os
import timm
from timm.loss import SoftTargetCrossEntropy
import pickle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import json
import os
import sys
import time
from pathlib import Path


# # Global

# In[7]:


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


# # Utils

# In[8]:


class TorchDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform=transform

    def __repr__(self):
        return str(self.hf_dataset)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        image = example['image']
        if self.transform:
            image =  self.transform(image)
        return image, example['label']


visualize = v2.Compose([v2.ToPILImage(), v2.Resize((100,100), antialias=False)])


try:
    train_dataset = pickle.load(open(TRAIN_DATA,"rb"))
    valid_dataset = pickle.load(open(VALID_DATA,"rb"))
except:
    train_dataset = load_dataset('Maysee/tiny-imagenet', split='train')
    valid_dataset = load_dataset('Maysee/tiny-imagenet', split='valid')
    pickle.dump(train_dataset, open(TRAIN_DATA,"wb"))
    pickle.dump(valid_dataset, open(VALID_DATA,"wb"))


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


train_loader = torch.utils.data.DataLoader(
    TorchDatasetWrapper(train_dataset, transform_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,  
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True
)

val_loader = torch.utils.data.DataLoader(
    TorchDatasetWrapper(valid_dataset, transform_valid),
    batch_size=3,
    shuffle=False,
    num_workers=2,  
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True
)
next(iter(val_loader))


# Mixup and CutMix
mixup_fn = timm.data.Mixup(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    cutmix_minmax=None,
    prob=0.5,  # Reduced probability to allow some original images
    switch_prob=0.5,
    mode='batch',
    label_smoothing=0.1,
    num_classes=NUM_CLASSES
)

import time
def benchmark_model(model, device, gpu=True, cpu=False, WARMUPS=10, batch_sizes=[1,2,4,8,16,32,64], n_iters=[50,25,10,10,10,10,10], acc=True, cpu_acc=False):
    def test_accuracy(model, device, WARMUPS=10, batch_sizes=[1,2,4,8,16,32,64], n_iters=[50,25,10,10,10,10,10], acc=True, cpu_acc=False):
        if not cpu_acc and device.type == "cpu":
            raise AssertionError("Set cpu_acc=True if you want to check accuracy with cpu, otherwise set acc=False to skip accuracy benchmark")
        model.to(device)
        test_loader = torch.utils.data.DataLoader(
            TorchDatasetWrapper(valid_dataset, transform_valid),
            batch_size=max(batch_sizes),
            shuffle=False,
            num_workers=2,  
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        test_loader_tqdm = tqdm(test_loader, desc=f"Testing Accuracy")
        total=0
        correct=0
        for images, labels in test_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(labels).sum().item()
            correct += batch_correct
            total += len(labels)
        return correct/total

    def test_gpu_speed(model, device, WARMUPS=10, batch_sizes=[1,2,4,8,16,32,64], n_iters=[50,25,10,10,10,10,10], acc=True, cpu_acc=False):
        print("Benchmarking GPU")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        results = []
        model.to(device)
        for batch_size,  n_iter in zip(batch_sizes, n_iters):
            test_loader = torch.utils.data.DataLoader(
                TorchDatasetWrapper(valid_dataset, transform_valid),
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,  
                pin_memory=True,
                prefetch_factor=4,
                persistent_workers=True
            )
            
            images, labels = next(iter(test_loader))
            images, labels = images.to(device), labels.to(device)
            for _ in range(WARMUPS):
                _ = model(images)
            print(f"Benchmarking batch size {batch_size}")
            torch.cuda.synchronize()
            start_event.record()
            with torch.no_grad():
                for _ in range(n_iter):
                    _ = model(images)
            end_event.record()
            torch.cuda.synchronize() 
            elapsed = start_event.elapsed_time(end_event) / 100
            avg_time_per_batch = elapsed / n_iter * 1000  # ms per batch
            avg_time_per_image = avg_time_per_batch / batch_size   # ms per image
            throughput = 1000 / avg_time_per_image         # images/second
            batch_result = {"bs":batch_size, "mode":"gpu",}
            batch_result["values"] = {
                "time_per_batch_ms": avg_time_per_batch,
                "time_per_image_ms": avg_time_per_image,
                "throughput_imgs/s": throughput
            }
            results.append(batch_result)
        return results
            
    def test_cpu_speed(model, device, WARMUPS=10, batch_sizes=[1,2,4,8,16,32,64], n_iters=[64,32,16,8,4,2,1], acc=True, cpu_acc=False):
        print("Benchmarking CPU")
        results = []
        model = model.to(torch.device("cpu"))
        for batch_size,  n_iter in zip(batch_sizes, n_iters):
            test_loader = torch.utils.data.DataLoader(
                TorchDatasetWrapper(valid_dataset, transform_valid),
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,  
                pin_memory=False,
                persistent_workers=False
            )
            images, labels = next(iter(test_loader))
            for _ in range(WARMUPS):
                _ = model(images)
            print(f"Benchmarking batch size {batch_size}")
            start_time = time.time()
            with torch.no_grad():
                for _ in range(n_iter):
                    _ = model(images)
            elapsed = time.time() - start_time
            avg_time_per_batch = elapsed / n_iter * 1000  # ms per batch
            avg_time_per_image = avg_time_per_batch / batch_size   # ms per image
            throughput = 1000 / avg_time_per_image         # images/second
            batch_result = {"bs":batch_size, "mode":"cpu",}
            batch_result["values"] = {
                "time_per_batch_ms": avg_time_per_batch,
                "time_per_image_ms": avg_time_per_image,
                "throughput_imgs/s": throughput
            }
            results.append(batch_result)
        return results

        def process_results(results_gpu, results_cpu):
            results = []
            if results_gpu and results_cpu: 
                results_zipped = zip(results_gpu, results_cpu)
            elif results_gpu:
                results_zipped = zip(results_gpu, [[]]*len(results_gpu))
            elif results_cpu:
                results_zipped = zip(results_cpu, [[]]*len(results_gpu))
        
            for result_gpu, result_cpu in results_zipped:
                dic = []
                if result_gpu: 
                    batch_size = result_gpu.pop("bs")
                    dic.append(result_gpu)
                if result_cpu: 
                    batch_size = result_cpu.pop("bs")
                    dic.append(result_cpu)
                results.append({"batch_size":batch_size,
                               "value":dic
                               })
            return results
        
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    if acc:
        accuracy = test_accuracy(model, device, WARMUPS, batch_sizes, n_iters, acc, cpu_acc)
    if gpu:
        results_gpu = test_gpu_speed(model, device, WARMUPS, batch_sizes, n_iters, acc, cpu_acc)
    if cpu:
        results_cpu = test_cpu_speed(model, device, WARMUPS, batch_sizes, [64,32,16,8,4,2,1], acc, cpu_acc)
    return process_results(results_gpu, results_cpu), accuracy


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Model Benchmarking Script')
    parser.add_argument('--model_weights', type=str, required=True,
                        help='Path to model weights file')
    parser.add_argument('--model_script', type=str, required=True,
                        help='Path to Python script defining the model')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Directory to save benchmark results')
    parser.add_argument('--cpu', action='store_true',
                        help='Benchmark on CPU')
    parser.add_argument('--gpu', action='store_true',
                        help='Benchmark on GPU')
    parser.add_argument('--accuracy', action='store_true',
                        help='Calculate model accuracy')
    
    args = parser.parse_args()

    # Import model definition
    model_dir = os.path.dirname(args.model_script)
    model_file = os.path.basename(args.model_script).rstrip('.py')
    sys.path.insert(0, model_dir)
    try:
        model_module = __import__(model_file)
        model = model_module.get_model(num_classes=NUM_CLASSES)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Load weights
    try:
        model.load_state_dict(torch.load(args.model_weights))
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    # Determine device
    device = torch.device('cpu')
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("Warning: CUDA not available, using CPU instead")
            args.gpu = False
            args.cpu = True

    # Create save directory if needed
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # Run benchmarking
    benchmark_results, accuracy = benchmark_model(
        model=model,
        device=device,
        gpu=args.gpu,
        cpu=args.cpu,
        acc=args.accuracy
    )

    # Save results
    results = {
        'hardware': str(device),
        'benchmark': benchmark_results,
        'accuracy': accuracy if args.accuracy else None
    }
    
    save_file = os.path.join(args.save_path, 'benchmark_results.json')
    with open(save_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {save_file}")

if __name__ == "__main__":
    main()



