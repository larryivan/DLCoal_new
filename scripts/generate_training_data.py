import numpy as np
import msprime
import h5py
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_simulation.demographic_models import piecewise_constant
from src.data_simulation.gaussian_process_models import generate_gp_history


def simulate_and_extract_genotypes(demography, sequence_length=1e6, num_samples=20, 
                                   recombination_rate=1e-8, mutation_rate=1e-8, 
                                   random_seed=None):
    """
    Simulates a tree sequence and extracts genotype matrix.
    
    Returns:
        genotype_matrix: numpy array of shape (num_haplotypes, num_sites)
        positions: numpy array of mutation positions
    """
    # Ensure random seed is valid for msprime (must be > 0 and < 2^32)
    if random_seed is None:
        random_seed = np.random.randint(1, 2**31)
    else:
        random_seed = max(1, min(random_seed, 2**31 - 1))
    
    # Simulate ancestry
    ts = msprime.sim_ancestry(
        samples=num_samples,  # This gives us 2*num_samples haplotypes (diploid)
        demography=demography,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=random_seed
    )
    
    # Add mutations
    ts = msprime.sim_mutations(
        ts, 
        rate=mutation_rate,
        random_seed=random_seed + 1  # Use different seed for mutations
    )
    
    # Extract genotype matrix
    # msprime returns a matrix of shape (num_sites, num_haplotypes)
    # We need to transpose it to get (num_haplotypes, num_sites)
    if ts.num_sites > 0:
        genotype_matrix = ts.genotype_matrix().T
        positions = ts.tables.sites.position
    else:
        # Handle case with no mutations
        genotype_matrix = np.array([]).reshape(ts.num_samples, 0)
        positions = np.array([])
    
    print(f"  Generated genotype matrix shape: {genotype_matrix.shape}")
    print(f"  Number of segregating sites: {ts.num_sites}")
    
    return genotype_matrix, positions


def discretize_ne_history(events, N0, time_points):
    """
    Convert event-based Ne history to values at fixed time points.
    
    Args:
        events: List of (time, size) tuples
        N0: Population size at time 0
        time_points: Array of time points where we want Ne values
        
    Returns:
        ne_values: Array of Ne values at the specified time points
    """
    ne_values = np.zeros(len(time_points))
    
    # Sort events by time
    sorted_events = sorted(events, key=lambda x: x[0])
    
    for i, t in enumerate(time_points):
        # Find the Ne value at time t
        ne_at_t = N0  # Default to N0
        
        for event_time, event_size in sorted_events:
            if event_time <= t:
                ne_at_t = event_size
            else:
                break
                
        ne_values[i] = ne_at_t
    
    return ne_values


def plot_ne_comparison(true_ne, time_points, sample_idx, save_dir):
    """
    Plot the true Ne history (we'll add predicted Ne later).
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(time_points, true_ne, 'b-', linewidth=2, label='True Ne')
    plt.xlabel('Time (generations ago)')
    plt.ylabel('Effective Population Size (Ne)')
    plt.title(f'Sample {sample_idx}: Ne History')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(save_dir, f'ne_comparison_sample_{sample_idx}.png'))
    plt.close()


def generate_dataset(num_samples, save_dir, max_variants=None):
    """
    Generate a dataset of genotype matrices and corresponding Ne histories.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Define time points where we'll evaluate Ne (log-spaced)
    time_points = np.logspace(1, 5, 100)  # 增加到100个时间点，从10到100,000代前
    
    # Lists to store data
    all_genotypes = []
    all_ne_histories = []
    max_sites = 0
    
    print(f"Generating {num_samples} samples...")
    
    for i in tqdm(range(num_samples)):
        # Generate a random demographic history
        events, N0 = generate_gp_history(
            random_state=i,
            add_sharp_bottleneck=(i % 3 == 0),  # Add bottleneck to every 3rd sample
            bottleneck_strength_reduction=0.05
        )
        
        # Create demography object
        demography = piecewise_constant(events=events, N0=N0)
        
        # Simulate genetic data with larger genome and more samples
        genotypes, positions = simulate_and_extract_genotypes(
            demography, 
            sequence_length=5e6,  # 增加基因组长度到5Mb
            num_samples=50,       # 增加样本数到50个个体（100个单倍型）
            random_seed=i
        )
        
        # Convert Ne history to fixed time points
        ne_history = discretize_ne_history(events, N0, time_points)
        
        # Store data
        all_genotypes.append(genotypes)
        all_ne_histories.append(ne_history)
        max_sites = max(max_sites, genotypes.shape[1])
        
        # Plot first few samples
        if i < 5:
            plot_ne_comparison(ne_history, time_points, i, save_dir)
    
    print(f"\nMaximum number of segregating sites: {max_sites}")
    
    # Pad all genotype matrices to the same width
    if max_variants is not None:
        pad_width = max_variants
    else:
        pad_width = max_sites
        
    padded_genotypes = []
    for genotypes in all_genotypes:
        if genotypes.shape[1] < pad_width:
            # Pad with zeros on the right
            pad_amount = pad_width - genotypes.shape[1]
            padded = np.pad(genotypes, ((0, 0), (0, pad_amount)), mode='constant')
        else:
            # Truncate if necessary
            padded = genotypes[:, :pad_width]
        padded_genotypes.append(padded)
    
    # Convert to numpy arrays
    genotype_array = np.array(padded_genotypes)
    ne_history_array = np.array(all_ne_histories)
    
    print(f"\nFinal dataset shapes:")
    print(f"  Genotypes: {genotype_array.shape}")
    print(f"  Ne histories: {ne_history_array.shape}")
    
    # Save to HDF5
    h5_path = os.path.join(save_dir, 'data.h5')
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('genotypes', data=genotype_array, compression='gzip')
        f.create_dataset('ne_histories', data=ne_history_array, compression='gzip')
        f.create_dataset('time_points', data=time_points)
        f.attrs['max_sites'] = max_sites
        f.attrs['num_samples'] = num_samples
    
    print(f"\nData saved to: {h5_path}")
    
    return genotype_array, ne_history_array, time_points


if __name__ == "__main__":
    # Generate large training and validation sets
    print("=== Generating training data ===")
    train_genotypes, train_ne, time_points = generate_dataset(
        num_samples=5000,  # 大幅增加训练样本数
        save_dir='data/train'
    )
    
    print("\n=== Generating validation data ===")
    val_genotypes, val_ne, _ = generate_dataset(
        num_samples=1000,  # 增加验证样本数
        save_dir='data/val',
        max_variants=train_genotypes.shape[2]  # Use same width as training
    )
    
    print("\n=== Generating test data ===")
    test_genotypes, test_ne, _ = generate_dataset(
        num_samples=500,   # 增加测试样本数
        save_dir='data/test',
        max_variants=train_genotypes.shape[2]  # Use same width as training
    ) 