import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import h5py
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import gc
from collections import defaultdict
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.deep_learning.mega_model import MegaDemographicCNN, MegaDemographicCNNLite

# 设置绘图样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
             print("MPS not available because the current PyTorch install was not "
                   "built with MPS enabled.")
             return torch.device("cpu")
        print("Apple MPS device detected.")
        return torch.device("mps")
    else:
        return torch.device("cpu")


class AdvancedTrainer:
    """Advanced trainer with comprehensive monitoring and visualization."""
    
    def __init__(self, model, device, save_dir='results'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = defaultdict(list)
        
        # Mixed precision training
        self.use_amp = (self.device.type != 'cpu')
        self.scaler = GradScaler(enabled=(self.device.type == 'cuda')) # GradScaler is only for CUDA
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def load_data(self, data_path):
        """Load data from HDF5 file with memory optimization."""
        print(f"📂 Loading data from {data_path}")
        
        with h5py.File(data_path, 'r') as f:
            genotypes = f['genotypes'][:]
            ne_histories = f['ne_histories'][:]
            time_points = f['time_points'][:]
            
            # Print data info
            print(f"   📊 Genotypes: {genotypes.shape} ({genotypes.dtype})")
            print(f"   📈 Ne histories: {ne_histories.shape} ({ne_histories.dtype})")
            print(f"   ⏰ Time points: {len(time_points)}")
            print(f"   💾 Memory: {genotypes.nbytes/1e9:.2f} GB + {ne_histories.nbytes/1e6:.1f} MB")
            
        return genotypes, ne_histories, time_points
    
    def create_data_loader(self, genotypes, ne_histories, batch_size=64, shuffle=True):
        """Create optimized PyTorch DataLoader."""
        # Convert to tensors
        genotypes_tensor = torch.FloatTensor(genotypes)
        ne_histories_tensor = torch.FloatTensor(np.log(ne_histories))  # Log space
        
        # Create dataset
        dataset = TensorDataset(genotypes_tensor, ne_histories_tensor)
        
        # Use more workers if not on Windows
        num_workers = min(os.cpu_count(), 8) if os.name != 'nt' else 0
        
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if self.device.type != 'cpu' else False,
            persistent_workers=(num_workers > 0)
        )
        
        return loader
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Train for one epoch with mixed precision."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"🏋️ Epoch {epoch}")
        
        for batch_genotypes, batch_ne in pbar:
            batch_genotypes = batch_genotypes.to(self.device, non_blocking=True)
            batch_ne = batch_ne.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(batch_genotypes)
                loss = criterion(outputs, batch_ne)
            
            if self.device.type == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg': f'{total_loss/num_batches:.6f}'
            })
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_genotypes, batch_ne in tqdm(val_loader, desc="🔍 Validating"):
                batch_genotypes = batch_genotypes.to(self.device, non_blocking=True)
                batch_ne = batch_ne.to(self.device, non_blocking=True)
                
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(batch_genotypes)
                    loss = criterion(outputs, batch_ne)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, num_epochs=200, learning_rate=0.001):
        """Complete training loop with advanced features."""
        print(f"🚀 Starting training with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Optimizer with advanced settings
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Validate
            val_loss = self.validate_epoch(val_loader, criterion)
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"📊 Epoch {epoch:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}, Time={epoch_time:.1f}s")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_model('best_model.pth')
                print(f"🎯 New best model! Val loss: {val_loss:.6f}")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
                self.plot_training_curves()
            
            # Early stopping check
            if epoch - self.best_epoch > 30:  # No improvement for 30 epochs
                print(f"⏹️ Early stopping at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        print(f"✅ Training completed in {total_time:.1f}s")
        print(f"🏆 Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
        
        return self.history
    
    def plot_training_curves(self):
        """Plot comprehensive training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Learning rate
        axes[0, 1].plot(epochs, self.history['lr'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Loss improvement
        if len(self.history['val_loss']) > 1:
            val_improvement = np.diff(self.history['val_loss'])
            axes[1, 0].plot(epochs[1:], val_improvement, 'purple', linewidth=1, alpha=0.7)
            axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Validation Loss Change')
            axes[1, 0].set_title('Validation Loss Improvement')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Training efficiency
        if len(epochs) > 10:
            window = 10
            train_smooth = np.convolve(self.history['train_loss'], np.ones(window)/window, mode='valid')
            val_smooth = np.convolve(self.history['val_loss'], np.ones(window)/window, mode='valid')
            smooth_epochs = epochs[window-1:]
            
            axes[1, 1].plot(smooth_epochs, train_smooth, 'b-', label='Train (smoothed)', linewidth=2)
            axes[1, 1].plot(smooth_epochs, val_smooth, 'r-', label='Val (smoothed)', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Smoothed Loss')
            axes[1, 1].set_title('Smoothed Loss Curves')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_predictions_and_plots(self, val_genotypes, val_ne_histories, time_points, max_plots=20):
        """Generate comprehensive prediction analysis."""
        print("🎨 Generating prediction analysis...")
        
        self.model.eval()
        
        # Convert to tensors
        val_genotypes_tensor = torch.FloatTensor(val_genotypes).to(self.device)
        
        with torch.no_grad():
            # Get predictions in batches to manage memory
            predictions_log = []
            batch_size = 32
            
            for i in range(0, len(val_genotypes), batch_size):
                batch = val_genotypes_tensor[i:i+batch_size]
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    batch_pred = self.model(batch)
                predictions_log.append(batch_pred.cpu().numpy())
            
            predictions_log = np.concatenate(predictions_log, axis=0)
            predictions = np.exp(predictions_log)  # Convert back from log space
        
        # Create comprehensive visualization
        self._create_prediction_plots(predictions, val_ne_histories, time_points, max_plots)
        
        # Calculate metrics
        mse = np.mean((np.log(predictions) - np.log(val_ne_histories))**2)
        mae = np.mean(np.abs(np.log(predictions) - np.log(val_ne_histories)))
        r2 = 1 - np.var(np.log(predictions) - np.log(val_ne_histories)) / np.var(np.log(val_ne_histories))
        
        metrics = {
            'mse_log': mse,
            'mae_log': mae,
            'r2_log': r2,
            'num_samples': len(predictions)
        }
        
        print(f"📊 Validation Metrics:")
        print(f"   MSE (log): {mse:.6f}")
        print(f"   MAE (log): {mae:.6f}")
        print(f"   R² (log): {r2:.6f}")
        
        # Save metrics
        with open(os.path.join(self.save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return predictions, metrics
    
    def _create_prediction_plots(self, predictions, val_ne_histories, time_points, max_plots):
        """Create detailed prediction visualization."""
        pred_dir = os.path.join(self.save_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        
        # Summary analysis
        self._create_summary_plots(predictions, val_ne_histories, time_points, pred_dir)
    
    def _create_summary_plots(self, predictions, val_ne_histories, time_points, save_dir):
        """Create summary analysis plots."""
        fig, axes = plt.subplots(3, 3, figsize=(25, 20))
        
        # Overall correlation
        all_true = val_ne_histories.flatten()
        all_pred = predictions.flatten()
        
        axes[0, 0].scatter(np.log(all_true), np.log(all_pred), alpha=0.3, s=1)
        min_val = min(np.log(all_true).min(), np.log(all_pred).min())
        max_val = max(np.log(all_true).max(), np.log(all_pred).max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        axes[0, 0].set_xlabel('Log(True Ne)')
        axes[0, 0].set_ylabel('Log(Predicted Ne)')
        axes[0, 0].set_title('Overall Correlation')
        axes[0, 0].grid(True, alpha=0.3)
        
        # More plots...
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self, filename):
        """Save model state."""
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': dict(self.history)
        }, path)
    
    def save_checkpoint(self, epoch):
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'history': dict(self.history),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }, path)


def main():
    # Set device
    device = get_device()
    print(f"🔥 Using device: {device}")
    
    # Create trainer and load data
    print("\n🏗️ Setting up large-scale training...")
    
    # For large dataset training
    trainer = AdvancedTrainer(None, device, save_dir='results/large_scale_run')
    
    # Load large datasets
    train_data_path = 'data/large_train/data.h5'
    val_data_path = 'data/large_val/data.h5'
    
    if not (os.path.exists(train_data_path) and os.path.exists(val_data_path)):
        print(f"❌ Large-scale data not found! Please run 'scripts/generate_mega_dataset.py' first.")
        return
        
    train_genotypes, train_ne_histories, time_points = trainer.load_data(train_data_path)
    val_genotypes, val_ne_histories, _ = trainer.load_data(val_data_path)
    
    # Model parameters
    num_haplotypes = train_genotypes.shape[1]
    max_variants = train_genotypes.shape[2]
    num_time_points = len(time_points)
    
    print(f"\n📐 Model parameters:")
    print(f"   Haplotypes: {num_haplotypes}")
    print(f"   Max variants: {max_variants:,}")
    print(f"   Time points: {num_time_points}")
    
    # Create the full model for large-scale training
    model = MegaDemographicCNN(
        num_haplotypes=num_haplotypes,
        max_variants=max_variants,
        num_time_points=num_time_points,
        hidden_dim=1024  # Use the full model capacity
    ).to(device)
    
    trainer.model = model
    
    print(f"🧠 Model: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders with a larger batch size
    train_loader = trainer.create_data_loader(train_genotypes, train_ne_histories, 
                                            batch_size=128, shuffle=True)
    val_loader = trainer.create_data_loader(val_genotypes, val_ne_histories, 
                                          batch_size=128, shuffle=False)
    
    # Train model for more epochs
    print("\n🚀 Starting large-scale training...")
    history = trainer.train(train_loader, val_loader, num_epochs=200, learning_rate=5e-4)
    
    # Generate predictions
    print("\n🎨 Generating final predictions on validation set...")
    predictions, metrics = trainer.generate_predictions_and_plots(
        val_genotypes, val_ne_histories, time_points, max_plots=20
    )
    
    print("\n\n✅✅✅ LARGE-SCALE TRAINING COMPLETED SUCCESSFULLY! ✅✅✅")
    print(f"   Find all results in the '{trainer.save_dir}' directory.")


if __name__ == "__main__":
    main() 