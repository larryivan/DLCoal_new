import numpy as np
import msprime
import h5py
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import gc

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_simulation.demographic_models import piecewise_constant
from src.data_simulation.gaussian_process_models import generate_gp_history


def simulate_single_sample(args):
    """
    Simulate a single sample (for multiprocessing).
    """
    i, sequence_length, num_samples, recombination_rate, mutation_rate = args
    
    try:
        # Generate a random demographic history
        events, N0 = generate_gp_history(
            random_state=i,
            add_sharp_bottleneck=(i % 3 == 0),  # Add bottleneck to every 3rd sample
            bottleneck_strength_reduction=0.05
        )
        
        # Create demography object
        demography = piecewise_constant(events=events, N0=N0)
        
        # Ensure random seed is valid for msprime
        random_seed = max(1, min(i + 1, 2**31 - 1))
        
        # Simulate ancestry
        ts = msprime.sim_ancestry(
            samples=num_samples,
            demography=demography,
            sequence_length=sequence_length,
            recombination_rate=recombination_rate,
            random_seed=random_seed
        )
        
        # Add mutations
        ts = msprime.sim_mutations(
            ts, 
            rate=mutation_rate,
            random_seed=random_seed + 1
        )
        
        # Extract genotype matrix
        if ts.num_sites > 0:
            genotype_matrix = ts.genotype_matrix().T
            positions = ts.tables.sites.position
        else:
            genotype_matrix = np.array([]).reshape(ts.num_samples, 0)
            positions = np.array([])
        
        return {
            'genotypes': genotype_matrix,
            'positions': positions,
            'events': events,
            'N0': N0,
            'sample_id': i,
            'num_sites': ts.num_sites
        }
        
    except Exception as e:
        print(f"Error in sample {i}: {e}")
        return None


def discretize_ne_history(events, N0, time_points):
    """
    Convert event-based Ne history to values at fixed time points.
    """
    ne_values = np.zeros(len(time_points))
    
    # Sort events by time
    sorted_events = sorted(events, key=lambda x: x[0])
    
    for i, t in enumerate(time_points):
        ne_at_t = N0  # Default to N0
        
        for event_time, event_size in sorted_events:
            if event_time <= t:
                ne_at_t = event_size
            else:
                break
                
        ne_values[i] = ne_at_t
    
    return ne_values


def generate_large_dataset(num_samples, save_dir, sequence_length=5e6, num_individuals=50,
                          recombination_rate=1e-8, mutation_rate=1e-8, 
                          batch_size=100, num_processes=None):
    """
    Generate a large dataset using multiprocessing and batch saving.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Define time points
    time_points = np.logspace(1, 5, 100)  # 100 time points from 10 to 100,000 generations ago
    
    if num_processes is None:
        num_processes = min(cpu_count(), 8)  # Limit to 8 processes to avoid memory issues
    
    print(f"Using {num_processes} processes for parallel simulation")
    print(f"Generating {num_samples} samples in batches of {batch_size}")
    
    # Prepare arguments for multiprocessing
    all_genotypes = []
    all_ne_histories = []
    max_sites = 0
    
    # Process in batches to manage memory
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_size_actual = batch_end - batch_start
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(num_samples-1)//batch_size + 1}")
        print(f"Samples {batch_start} to {batch_end-1}")
        
        # Prepare arguments for this batch
        args_list = []
        for i in range(batch_start, batch_end):
            args_list.append((i, sequence_length, num_individuals, recombination_rate, mutation_rate))
        
        # Simulate in parallel
        with Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap(simulate_single_sample, args_list),
                total=batch_size_actual,
                desc="Simulating"
            ))
        
        # Process results
        batch_genotypes = []
        batch_ne_histories = []
        
        for result in results:
            if result is not None:
                # Convert Ne history to fixed time points
                ne_history = discretize_ne_history(result['events'], result['N0'], time_points)
                
                batch_genotypes.append(result['genotypes'])
                batch_ne_histories.append(ne_history)
                max_sites = max(max_sites, result['genotypes'].shape[1])
                
                # Print progress info
                if result['sample_id'] % 500 == 0:
                    print(f"  Sample {result['sample_id']}: {result['num_sites']} sites, "
                          f"genotype shape {result['genotypes'].shape}")
        
        all_genotypes.extend(batch_genotypes)
        all_ne_histories.extend(batch_ne_histories)
        
        # Clear memory
        del batch_genotypes, batch_ne_histories, results
        gc.collect()
    
    print(f"\nCompleted simulation. Maximum number of segregating sites: {max_sites}")
    print(f"Total samples generated: {len(all_genotypes)}")
    
    # Pad all genotype matrices to the same width
    print("Padding genotype matrices...")
    padded_genotypes = []
    
    for i, genotypes in enumerate(tqdm(all_genotypes, desc="Padding")):
        if genotypes.shape[1] < max_sites:
            # Pad with zeros on the right
            pad_amount = max_sites - genotypes.shape[1]
            padded = np.pad(genotypes, ((0, 0), (0, pad_amount)), mode='constant')
        else:
            padded = genotypes
        padded_genotypes.append(padded)
    
    # Convert to numpy arrays
    print("Converting to numpy arrays...")
    genotype_array = np.array(padded_genotypes, dtype=np.int8)  # Use int8 to save memory
    ne_history_array = np.array(all_ne_histories, dtype=np.float32)  # Use float32 to save memory
    
    print(f"\nFinal dataset shapes:")
    print(f"  Genotypes: {genotype_array.shape} (dtype: {genotype_array.dtype})")
    print(f"  Ne histories: {ne_history_array.shape} (dtype: {ne_history_array.dtype})")
    print(f"  Memory usage: {genotype_array.nbytes / 1e9:.2f} GB (genotypes) + "
          f"{ne_history_array.nbytes / 1e6:.2f} MB (Ne histories)")
    
    # Save to HDF5 with compression
    h5_path = os.path.join(save_dir, 'data.h5')
    print(f"Saving to {h5_path}...")
    
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('genotypes', data=genotype_array, 
                        compression='gzip', compression_opts=9, shuffle=True)
        f.create_dataset('ne_histories', data=ne_history_array, 
                        compression='gzip', compression_opts=9)
        f.create_dataset('time_points', data=time_points)
        f.attrs['max_sites'] = max_sites
        f.attrs['num_samples'] = len(all_genotypes)
        f.attrs['sequence_length'] = sequence_length
        f.attrs['num_individuals'] = num_individuals
    
    print(f"Data saved to: {h5_path}")
    
    # Plot first few samples
    print("Generating sample plots...")
    for i in range(min(5, len(all_ne_histories))):
        plt.figure(figsize=(10, 6))
        plt.loglog(time_points, all_ne_histories[i], 'b-', linewidth=2, label='True Ne')
        plt.xlabel('Time (generations ago)')
        plt.ylabel('Effective Population Size (Ne)')
        plt.title(f'Sample {i}: Ne History')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(os.path.join(save_dir, f'ne_history_sample_{i}.png'))
        plt.close()
    
    return genotype_array, ne_history_array, time_points


if __name__ == "__main__":
    print("=== 生成大规模训练数据集 ===")
    print("参数设置：")
    print("  - 基因组长度: 5 Mb")
    print("  - 每个样本个体数: 50 (100个单倍型)")
    print("  - 时间点数: 100")
    print("  - 训练样本数: 10,000")
    print("  - 验证样本数: 2,000") 
    print("  - 测试样本数: 1,000")
    
    # Generate training data
    train_genotypes, train_ne, time_points = generate_large_dataset(
        num_samples=10000,  # 大规模训练集
        save_dir='data/train',
        batch_size=200,     # 批量处理以管理内存
        num_processes=6     # 并行进程数
    )
    
    print("\n=== 生成验证数据集 ===")
    val_genotypes, val_ne, _ = generate_large_dataset(
        num_samples=2000,   # 大规模验证集
        save_dir='data/val',
        batch_size=200,
        num_processes=6
    )
    
    print("\n=== 生成测试数据集 ===")
    test_genotypes, test_ne, _ = generate_large_dataset(
        num_samples=1000,   # 测试集
        save_dir='data/test',
        batch_size=200,
        num_processes=6
    )
    
    print("\n=== 数据生成完成！===")
    print(f"总数据量: 训练集 {len(train_genotypes)} + 验证集 {len(val_genotypes)} + 测试集 {len(test_genotypes)} = {len(train_genotypes) + len(val_genotypes) + len(test_genotypes)} 样本") 