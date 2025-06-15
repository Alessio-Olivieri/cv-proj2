import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional, Dict, Any, Union, List

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: torch.nn.Module,
        val_criterion: torch.nn.Module,
        scheduler: _LRScheduler,
        writer: SummaryWriter,
        device: torch.device,
        scaler: torch.cuda.amp.GradScaler,
        num_epochs: int,
        log_interval: int,
        model_dir: str,
        early_stop_patience: int,
        checkpoint_interval: int = None,
        mixup_fn: Optional[callable] = None,
        model_name: str = "interpolated_vit_tiny_imagenet.pth",
        resume: bool = False  
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.val_criterion=val_criterion
        self.scheduler = scheduler
        self.writer = writer
        self.device = device
        self.scaler = scaler
        self.num_epochs = num_epochs
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval
        self.model_dir = model_dir
        self.early_stop_patience = early_stop_patience
        self.mixup_fn = mixup_fn
        self.model_name = model_name
        
        self.start_epoch = 0
        self.best_acc = 0.0

        if resume:
            checkpoint_path = os.path.join(self.model_dir, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                print(f"Resuming training from checkpoint: {checkpoint_path}")
                self.load_checkpoint(checkpoint_path)
            else:
                print(f"Resume flag is set, but no checkpoint was found at {checkpoint_path}. Starting from scratch.")

    def load_checkpoint(self, path: str):
        """Loads training state from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state, handling DataParallel wrapper if necessary
        model_state_dict = checkpoint['state_dict']
        # If the model was saved with DataParallel, it will have a 'module.' prefix.
        # This handles loading it into a non-DataParallel model.
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.load_state_dict(model_state_dict)
        else:
            # Create a new state_dict without the 'module.' prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                if k.startswith('module.'):
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.start_epoch = checkpoint['epoch']
        self.best_acc = checkpoint['best_acc']
        
        print(f"Checkpoint loaded. Resuming from epoch {self.start_epoch + 1} with best accuracy {self.best_acc:.2f}%.")


    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        train_loader_tqdm = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.num_epochs} [Training]", 
            leave=False
        )
        
        for batch_idx, (images, labels) in enumerate(train_loader_tqdm):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.mixup_fn:
                images, labels = self.mixup_fn(images, labels)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type):
                outputs, _ = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = loss.item()
            _, predicted = outputs.max(1)
            targets = labels.argmax(1) if self.mixup_fn else labels
            batch_correct = predicted.eq(targets).sum().item()
            batch_total = labels.size(0)
            batch_acc = 100.0 * batch_correct / batch_total

            running_loss += batch_loss * batch_total
            correct += batch_correct
            total += batch_total

            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Loss/train_batch', batch_loss, global_step)
            self.writer.add_scalar('Accuracy/train_batch', batch_acc, global_step)

            if (batch_idx + 1) % self.log_interval == 0 or (batch_idx + 1) == len(self.train_loader):
                cumulative_loss = running_loss / total
                cumulative_acc = 100.0 * correct / total
                train_loader_tqdm.set_postfix(
                    loss=f"{cumulative_loss:.4f}", 
                    accuracy=f"{cumulative_acc:.2f}%"
                )

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def validate(self, epoch: int) -> float:
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0

        val_loader_tqdm = tqdm(
            self.val_loader, 
            desc=f"Epoch {epoch+1}/{self.num_epochs} [Validation]", 
            leave=False
        )

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader_tqdm):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self.device.type):
                    outputs, _ = self.model(images)
                    loss = self.val_criterion(outputs, labels)

                batch_loss = loss.item()
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                batch_total = labels.size(0)

                val_loss += batch_loss * batch_total
                correct += batch_correct
                total += batch_total

                global_step = epoch * len(self.train_loader) + len(self.train_loader) + batch_idx
                batch_acc = 100.0 * batch_correct / batch_total
                self.writer.add_scalar('Loss/val_batch', batch_loss, global_step)
                self.writer.add_scalar('Accuracy/val_batch', batch_acc, global_step)

                if (batch_idx + 1) % self.log_interval == 0 or (batch_idx + 1) == len(self.val_loader):
                    cumulative_loss = val_loss / total
                    cumulative_acc = 100.0 * correct / total
                    val_loader_tqdm.set_postfix(
                        loss=f"{cumulative_loss:.4f}", 
                        accuracy=f"{cumulative_acc:.2f}%"
                    )

        epoch_loss = val_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def save_model(self, epoch: int, is_best: bool = False) -> None:
        """Save current model state"""
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) 
                          else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'scaler': self.scaler.state_dict()
        }
        os.makedirs(self.model_dir, exist_ok=True) # Ensure directory exists
        torch.save(state, os.path.join(self.model_dir, 'checkpoint.pth'))
        if is_best:
            torch.save(state, os.path.join(self.model_dir, self.model_name))

    def train(self) -> float:
        print(f"Logging to {self.writer.log_dir}")
        os.makedirs(self.model_dir, exist_ok=True)
        
        no_improv_epochs = 0
        smallest_val_loss = float("inf")
        
        for epoch in range(self.start_epoch, self.num_epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            print(f"Epoch [{epoch+1}/{self.num_epochs}] | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            self.scheduler.step()

            # Save model checkpoint after every epoch
            self.save_model(epoch)

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                # Save the best model separately
                self.save_model(epoch, is_best=True)
                print(f"New best model saved with accuracy: {self.best_acc:.2f}%")

            if val_loss < smallest_val_loss:
                no_improv_epochs = 0
                smallest_val_loss = val_loss
            else:
                no_improv_epochs+=1

            if self.checkpoint_interval and (epoch + 1) % self.checkpoint_interval == 0:
                print(f"Saving periodic checkpoint at epoch {epoch + 1}")
            
            if no_improv_epochs >= self.early_stop_patience:
                print(f"Early stopping triggered after {self.early_stop_patience} epochs with no improvement.")
                break

        print(f"Training complete. Best validation accuracy: {self.best_acc:.2f}%")
        self.writer.close()
        return self.best_acc