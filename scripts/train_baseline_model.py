import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.deep_learning.models import CNN1D_Baseline, CNN1D_Deeper, CNN1D_Large


def load_data(data_path):
    """Load data from HDF5 file."""
    with h5py.File(data_path, 'r') as f:
        genotypes = f['genotypes'][:]
        ne_histories = f['ne_histories'][:]
        time_points = f['time_points'][:]
        
    print(f"Loaded data from {data_path}")
    print(f"  Genotypes shape: {genotypes.shape}")
    print(f"  Ne histories shape: {ne_histories.shape}")
    print(f"  Time points shape: {time_points.shape}")
    
    return genotypes, ne_histories, time_points


def create_data_loader(genotypes, ne_histories, batch_size=16, shuffle=True):
    """Create PyTorch DataLoader."""
    # Convert to tensors
    genotypes_tensor = torch.FloatTensor(genotypes)
    ne_histories_tensor = torch.FloatTensor(np.log(ne_histories))  # Use log(Ne)
    
    # Create dataset
    dataset = TensorDataset(genotypes_tensor, ne_histories_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cpu'):
    """Train the model."""
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    print(f"Training on device: {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_genotypes, batch_ne in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_genotypes = batch_genotypes.to(device)
            batch_ne = batch_ne.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_genotypes)
            loss = criterion(outputs, batch_ne)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_genotypes, batch_ne in val_loader:
                batch_genotypes = batch_genotypes.to(device)
                batch_ne = batch_ne.to(device)
                
                outputs = model(batch_genotypes)
                loss = criterion(outputs, batch_ne)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    return train_losses, val_losses


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def generate_predictions_and_plots(model, val_genotypes, val_ne_histories, time_points, 
                                   save_dir, device='cpu', max_plots=10):
    """Generate predictions and create comparison plots."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to tensors
    val_genotypes_tensor = torch.FloatTensor(val_genotypes).to(device)
    
    with torch.no_grad():
        # Get predictions for all validation samples
        predictions_log = model(val_genotypes_tensor).cpu().numpy()
        predictions = np.exp(predictions_log)  # Convert back from log space
    
    # Create individual plots for first few samples
    num_plots = min(max_plots, len(val_genotypes))
    
    for i in range(num_plots):
        plt.figure(figsize=(12, 8))
        
        # Plot true vs predicted Ne
        plt.subplot(2, 1, 1)
        plt.loglog(time_points, val_ne_histories[i], 'b-', linewidth=3, label='True Ne', alpha=0.8)
        plt.loglog(time_points, predictions[i], 'r--', linewidth=2, label='Predicted Ne', alpha=0.8)
        plt.xlabel('Time (generations ago)')
        plt.ylabel('Effective Population Size (Ne)')
        plt.title(f'Validation Sample {i}: Ne History Comparison')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # Plot residuals (log scale)
        plt.subplot(2, 1, 2)
        residuals = np.log(predictions[i]) - np.log(val_ne_histories[i])
        plt.semilogx(time_points, residuals, 'g-', linewidth=2, alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Time (generations ago)')
        plt.ylabel('Log(Predicted) - Log(True)')
        plt.title('Prediction Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_comparison_sample_{i}.png'), dpi=300)
        plt.close()
    
    # Create a summary plot with multiple samples
    plt.figure(figsize=(15, 10))
    
    # Plot multiple samples on the same plot
    colors = plt.cm.tab10(np.linspace(0, 1, min(5, num_plots)))
    
    for i in range(min(5, num_plots)):
        plt.subplot(2, 3, i+1)
        plt.loglog(time_points, val_ne_histories[i], 'b-', linewidth=2, label='True', alpha=0.8)
        plt.loglog(time_points, predictions[i], 'r--', linewidth=2, label='Predicted', alpha=0.8)
        plt.title(f'Sample {i}')
        plt.xlabel('Time (gen ago)')
        plt.ylabel('Ne')
        if i == 0:
            plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Overall correlation plot
    plt.subplot(2, 3, 6)
    all_true = val_ne_histories[:num_plots].flatten()
    all_pred = predictions[:num_plots].flatten()
    
    plt.scatter(np.log(all_true), np.log(all_pred), alpha=0.5, s=10)
    min_val = min(np.log(all_true).min(), np.log(all_pred).min())
    max_val = max(np.log(all_true).max(), np.log(all_pred).max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    plt.xlabel('Log(True Ne)')
    plt.ylabel('Log(Predicted Ne)')
    plt.title('Overall Correlation')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_summary.png'), dpi=300)
    plt.close()
    
    # Calculate and print metrics
    mse = np.mean((np.log(predictions) - np.log(val_ne_histories))**2)
    mae = np.mean(np.abs(np.log(predictions) - np.log(val_ne_histories)))
    
    print(f"\nValidation Metrics:")
    print(f"  MSE (log scale): {mse:.6f}")
    print(f"  MAE (log scale): {mae:.6f}")
    
    return predictions


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading training data...")
    train_genotypes, train_ne_histories, time_points = load_data('data/train/data.h5')
    
    print("Loading validation data...")
    val_genotypes, val_ne_histories, _ = load_data('data/val/data.h5')
    
    # Create data loaders with larger batch sizes for big dataset
    train_loader = create_data_loader(train_genotypes, train_ne_histories, batch_size=32, shuffle=True)
    val_loader = create_data_loader(val_genotypes, val_ne_histories, batch_size=32, shuffle=False)
    
    # Model parameters
    num_haplotypes = train_genotypes.shape[1]  # Should be 2 * num_samples
    max_variants = train_genotypes.shape[2]
    num_time_points = len(time_points)
    
    print(f"\nModel parameters:")
    print(f"  Number of haplotypes: {num_haplotypes}")
    print(f"  Maximum variants: {max_variants}")
    print(f"  Number of time points: {num_time_points}")
    
    # Create large model for big dataset
    model = CNN1D_Large(
        num_haplotypes=num_haplotypes,
        max_variants=max_variants,
        num_time_points=num_time_points,
        hidden_dim=512  # 更大的隐藏层维度
    )
    
    # Train model with more epochs for larger dataset
    print("\nStarting training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=100, learning_rate=0.001, device=device  # 增加训练轮数
    )
    
    # Plot training curves
    os.makedirs('results', exist_ok=True)
    plot_training_curves(train_losses, val_losses, 'results/training_curves.png')
    
    # Generate predictions and plots
    print("\nGenerating prediction plots...")
    predictions = generate_predictions_and_plots(
        model, val_genotypes, val_ne_histories, time_points,
        save_dir='results/predictions', device=device, max_plots=10
    )
    
    # Save model
    torch.save(model.state_dict(), 'results/baseline_model.pth')
    print("\nModel saved to results/baseline_model.pth")
    
    print("\nTraining completed! Check the 'results' directory for outputs.")


if __name__ == "__main__":
    main() 