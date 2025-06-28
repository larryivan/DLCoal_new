import numpy as np
import msprime
import h5py
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
import gc
import time
from functools import partial

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_simulation.demographic_models import piecewise_constant
from src.data_simulation.gaussian_process_models import generate_gp_history

# 设置绘图样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def simulate_single_sample(args):
    """
    Simulate a single sample with enhanced demographic diversity.
    """
    i, sequence_length, num_samples, recombination_rate, mutation_rate, enhanced_diversity = args
    
    try:
        # Generate diverse demographic histories
        if enhanced_diversity:
            # 更复杂的人口历史模式
            events, N0 = generate_gp_history(
                random_state=i,
                add_sharp_bottleneck=(i % 2 == 0),  # 50%概率瓶颈
                bottleneck_strength_reduction=np.random.uniform(0.01, 0.1),  # 随机瓶颈强度
                max_time=200000,  # 更大的时间范围
                num_control_points=np.random.randint(3, 8),  # 3-7个控制点
                num_discretization_points=150  # 更高分辨率
            )
        else:
            events, N0 = generate_gp_history(
                random_state=i,
                add_sharp_bottleneck=(i % 3 == 0),
                bottleneck_strength_reduction=0.05,
                max_time=100000
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
            'num_sites': ts.num_sites,
            'tree_height': ts.max_root_time if ts.num_trees > 0 else 0
        }
        
    except Exception as e:
        print(f"Error in sample {i}: {e}")
        return None


def discretize_ne_history(events, N0, time_points):
    """Convert event-based Ne history to values at fixed time points."""
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


def create_comprehensive_plots(all_ne_histories, time_points, save_dir, dataset_name):
    """Create comprehensive visualization of the dataset."""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Creating comprehensive plots for {dataset_name}...")
    
    # 1. Ne历史分布概览
    plt.figure(figsize=(20, 12))
    
    # 子图1: 多个Ne历史轨迹
    plt.subplot(2, 3, 1)
    n_plot = min(50, len(all_ne_histories))
    for i in range(n_plot):
        plt.loglog(time_points, all_ne_histories[i], alpha=0.3, linewidth=0.8)
    plt.xlabel('Time (generations ago)')
    plt.ylabel('Effective Population Size (Ne)')
    plt.title(f'{dataset_name}: Ne Trajectories (n={n_plot})')
    plt.grid(True, alpha=0.3)
    
    # 子图2: Ne值分布直方图
    plt.subplot(2, 3, 2)
    all_ne_flat = np.log10(np.array(all_ne_histories).flatten())
    plt.hist(all_ne_flat, bins=50, alpha=0.7, density=True)
    plt.xlabel('log10(Ne)')
    plt.ylabel('Density')
    plt.title('Ne Distribution')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 不同时间点的Ne分布
    plt.subplot(2, 3, 3)
    time_indices = [0, len(time_points)//4, len(time_points)//2, 3*len(time_points)//4, -1]
    for idx in time_indices:
        ne_at_time = [hist[idx] for hist in all_ne_histories]
        plt.hist(np.log10(ne_at_time), bins=30, alpha=0.6, 
                label=f't={time_points[idx]:.0f}', density=True)
    plt.xlabel('log10(Ne)')
    plt.ylabel('Density')
    plt.title('Ne Distribution at Different Times')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4: Ne变化幅度分析
    plt.subplot(2, 3, 4)
    ne_ranges = []
    ne_means = []
    for hist in all_ne_histories:
        ne_ranges.append(np.log10(np.max(hist)) - np.log10(np.min(hist)))
        ne_means.append(np.log10(np.mean(hist)))
    
    plt.scatter(ne_means, ne_ranges, alpha=0.6, s=10)
    plt.xlabel('Mean log10(Ne)')
    plt.ylabel('log10(Ne) Range')
    plt.title('Ne Variability vs Mean')
    plt.grid(True, alpha=0.3)
    
    # 子图5: 时间序列相关性热图
    plt.subplot(2, 3, 5)
    if len(all_ne_histories) > 10:
        sample_histories = np.array(all_ne_histories[:100])  # 取前100个样本
        correlation_matrix = np.corrcoef(sample_histories)
        im = plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.title('Sample Correlation Matrix')
    
    # 子图6: Ne斜率分析
    plt.subplot(2, 3, 6)
    slopes = []
    for hist in all_ne_histories:
        log_ne = np.log(hist)
        log_time = np.log(time_points)
        slope = np.polyfit(log_time, log_ne, 1)[0]
        slopes.append(slope)
    
    plt.hist(slopes, bins=50, alpha=0.7, density=True)
    plt.xlabel('Ne Trajectory Slope')
    plt.ylabel('Density')
    plt.title('Distribution of Ne Slopes')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_comprehensive_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 详细的单个样本展示
    plt.figure(figsize=(25, 15))
    n_detailed = min(12, len(all_ne_histories))
    
    for i in range(n_detailed):
        plt.subplot(3, 4, i+1)
        plt.loglog(time_points, all_ne_histories[i], 'b-', linewidth=2, alpha=0.8)
        plt.xlabel('Time (gen ago)')
        plt.ylabel('Ne')
        plt.title(f'Sample {i}')
        plt.grid(True, which="both", alpha=0.3)
        
        # 添加统计信息
        ne_min = np.min(all_ne_histories[i])
        ne_max = np.max(all_ne_histories[i])
        plt.text(0.05, 0.95, f'Range: {ne_min:.0f}-{ne_max:.0f}', 
                transform=plt.gca().transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.suptitle(f'{dataset_name}: Detailed Sample Trajectories', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_detailed_samples.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def generate_mega_dataset(num_samples, save_dir, sequence_length=10e6, num_individuals=100,
                         recombination_rate=1e-8, mutation_rate=1e-8, 
                         batch_size=500, num_processes=None, enhanced_diversity=True):
    """
    Generate mega-scale dataset optimized for 104-core server.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 超高分辨率时间点
    time_points = np.logspace(1, 5.5, 200)  # 200个时间点，从10到316,228代前
    
    if num_processes is None:
        num_processes = min(cpu_count(), 100)  # 利用104核心
    
    print(f"🚀 MEGA DATASET GENERATION")
    print(f"📊 Dataset: {num_samples:,} samples")
    print(f"🧬 Genome: {sequence_length/1e6:.1f} Mb, {num_individuals} individuals ({2*num_individuals} haplotypes)")
    print(f"⏰ Time resolution: {len(time_points)} points")
    print(f"💻 Using {num_processes} processes")
    print(f"🔄 Batch size: {batch_size}")
    
    start_time = time.time()
    
    # Prepare arguments for multiprocessing
    all_genotypes = []
    all_ne_histories = []
    max_sites = 0
    total_sites = 0
    
    # Process in batches to manage memory
    num_batches = (num_samples - 1) // batch_size + 1
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_samples)
        batch_size_actual = batch_end - batch_start
        
        print(f"\n📦 Batch {batch_idx+1}/{num_batches}")
        print(f"   Samples {batch_start:,} to {batch_end-1:,}")
        
        # Prepare arguments for this batch
        simulate_func = partial(simulate_single_sample)
        args_list = []
        for i in range(batch_start, batch_end):
            args_list.append((i, sequence_length, num_individuals, 
                            recombination_rate, mutation_rate, enhanced_diversity))
        
        # Simulate in parallel
        batch_start_time = time.time()
        with Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap(simulate_func, args_list),
                total=batch_size_actual,
                desc="🧬 Simulating"
            ))
        
        batch_time = time.time() - batch_start_time
        print(f"   ⏱️  Batch time: {batch_time:.1f}s ({batch_time/batch_size_actual:.2f}s per sample)")
        
        # Process results
        batch_genotypes = []
        batch_ne_histories = []
        batch_sites = []
        
        for result in results:
            if result is not None:
                # Convert Ne history to fixed time points
                ne_history = discretize_ne_history(result['events'], result['N0'], time_points)
                
                batch_genotypes.append(result['genotypes'])
                batch_ne_histories.append(ne_history)
                batch_sites.append(result['num_sites'])
                max_sites = max(max_sites, result['genotypes'].shape[1])
                total_sites += result['num_sites']
        
        if batch_sites:  # 只有在有有效样本时才打印统计信息
            print(f"   📈 Sites: avg={np.mean(batch_sites):.0f}, max={np.max(batch_sites)}, total_max={max_sites}")
            print(f"   ✅ Valid samples: {len(batch_genotypes)}/{batch_size_actual}")
        else:
            print(f"   ⚠️  No valid samples generated in this batch!")
        
        all_genotypes.extend(batch_genotypes)
        all_ne_histories.extend(batch_ne_histories)
        
        # Memory cleanup
        del batch_genotypes, batch_ne_histories, results, batch_sites
        gc.collect()
    
    total_time = time.time() - start_time
    print(f"\n✅ Simulation completed!")
    print(f"   ⏱️  Total time: {total_time:.1f}s ({total_time/num_samples:.2f}s per sample)")
    print(f"   📊 Generated {len(all_genotypes):,} samples")
    
    if len(all_genotypes) == 0:
        print("❌ No valid samples generated! Please check the simulation parameters.")
        return None, None, None
    
    print(f"   🧬 Max segregating sites: {max_sites:,}")
    print(f"   📈 Avg sites per sample: {total_sites/len(all_genotypes):.0f}")
    
    # Create comprehensive plots
    create_comprehensive_plots(all_ne_histories, time_points, save_dir, 
                             f"Dataset_{len(all_genotypes)}")
    
    # Pad genotype matrices
    print("\n🔧 Padding genotype matrices...")
    padded_genotypes = []
    
    for i, genotypes in enumerate(tqdm(all_genotypes, desc="Padding")):
        if genotypes.shape[1] < max_sites:
            pad_amount = max_sites - genotypes.shape[1]
            padded = np.pad(genotypes, ((0, 0), (0, pad_amount)), mode='constant')
        else:
            padded = genotypes
        padded_genotypes.append(padded)
    
    # Convert to optimized arrays
    print("🔄 Converting to numpy arrays...")
    genotype_array = np.array(padded_genotypes, dtype=np.int8)
    ne_history_array = np.array(all_ne_histories, dtype=np.float32)
    
    print(f"\n📏 Final dataset dimensions:")
    print(f"   Genotypes: {genotype_array.shape} ({genotype_array.dtype})")
    print(f"   Ne histories: {ne_history_array.shape} ({ne_history_array.dtype})")
    print(f"   💾 Memory: {genotype_array.nbytes/1e9:.2f} GB + {ne_history_array.nbytes/1e6:.1f} MB")
    
    # Save with maximum compression
    h5_path = os.path.join(save_dir, 'data.h5')
    print(f"💾 Saving to {h5_path}...")
    
    save_start = time.time()
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('genotypes', data=genotype_array, 
                        compression='gzip', compression_opts=9, 
                        shuffle=True, fletcher32=True)
        f.create_dataset('ne_histories', data=ne_history_array, 
                        compression='gzip', compression_opts=9, fletcher32=True)
        f.create_dataset('time_points', data=time_points)
        
        # Metadata
        f.attrs['max_sites'] = max_sites
        f.attrs['num_samples'] = len(all_genotypes)
        f.attrs['sequence_length'] = sequence_length
        f.attrs['num_individuals'] = num_individuals
        f.attrs['generation_time'] = total_time
        f.attrs['enhanced_diversity'] = enhanced_diversity
    
    save_time = time.time() - save_start
    file_size = os.path.getsize(h5_path) / 1e9
    print(f"✅ Saved! File size: {file_size:.2f} GB (save time: {save_time:.1f}s)")
    
    return genotype_array, ne_history_array, time_points


if __name__ == "__main__":
    print("🎯 MEGA-SCALE DEMOGRAPHIC INFERENCE DATASET")
    print("=" * 60)
    
    # =========================================================================
    # ==                 MEGA-SCALE DATASET CONFIGURATION                    ==
    # =========================================================================
    
    # 训练集
    print("\n🚀 PHASE 1: Generating Mega-Scale Training Set")
    train_data = generate_mega_dataset(
        num_samples=100000,
        save_dir='data/large_train',
        sequence_length=20e6,      # 20 Mb
        num_individuals=200,       # 400 haplotypes
        batch_size=1000,           # Process 1000 samples per batch
        num_processes=min(100, cpu_count()) # Use up to 100 cores
    )
    
    if train_data[0] is None:
        print("❌ Training data generation failed. Aborting.")
        sys.exit(1)

    # 验证集
    print("\n🚀 PHASE 2: Generating Mega-Scale Validation Set")
    val_data = generate_mega_dataset(
        num_samples=10000,
        save_dir='data/large_val',
        sequence_length=20e6,
        num_individuals=200,
        batch_size=1000,
        num_processes=min(100, cpu_count())
    )
    
    # 测试集
    print("\n🚀 PHASE 3: Generating Mega-Scale Test Set")
    test_data = generate_mega_dataset(
        num_samples=5000,
        save_dir='data/large_test',
        sequence_length=20e6,
        num_individuals=200,
        batch_size=1000,
        num_processes=min(100, cpu_count())
    )

    print("\n\n✅✅✅ ALL MEGA-SCALE DATASETS GENERATED SUCCESSFULLY! ✅✅✅")
    total_samples = train_data[0].shape[0] + val_data[0].shape[0] + test_data[0].shape[0]
    print(f"   Total samples: {total_samples:,}")
    print("   Ready for large-scale training!") 