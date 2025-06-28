import msprime
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the demographic models we created
from src.data_simulation.demographic_models import (
    constant_size,
    single_bottleneck,
    single_expansion,
    exponential_growth,
    piecewise_constant
)
# Import the new GP model generator
from src.data_simulation.gaussian_process_models import generate_gp_history

def plot_demography(demography, log_time=True):
    """
    Plots the population size history from a msprime.Demography object.
    This version uses the recommended msprime API for robustness.

    Args:
        demography (msprime.Demography): The demography to plot.
        log_time (bool): Whether to plot time on a logarithmic scale.
    """
    fig, ax = plt.subplots()

    # Assuming a single population model for this plotting function
    pop = demography.populations[0]
    pop_name = pop.name
    print(f"--- Plotting Demography for: {pop_name} ---")
    print("   Manually constructing history from demography events.")

    # Manually construct the history from the demography's events list.
    # This is more robust than using the debugger's internal properties.
    events = sorted(
        [e for e in demography.events if isinstance(e, msprime.PopulationParametersChange)],
        key=lambda e: e.time
    )

    # Start with the size at time 0
    times = [0]
    sizes = [pop.initial_size]

    # For each event, add points to create the step-like plot
    for event in events:
        # Add a point just before the event to create the horizontal line
        times.append(event.time)
        sizes.append(sizes[-1])
        # Add a point at the event time with the new size for the vertical drop/rise
        times.append(event.time)
        sizes.append(event.initial_size)

    # Add a final point to extend the plot to the right, for visualization
    final_time = times[-1] * 1.5 if times[-1] > 0 else 100000
    if len(times) == 1: # Handle constant population size
        final_time = 100000
    times.append(final_time)
    sizes.append(sizes[-1])

    print(f"Constructed times for plotting: {times}")
    print(f"Constructed sizes for plotting: {sizes}")
    
    # For log scale, we cannot plot time=0, so we replace it with a small number.
    if log_time and times[0] == 0:
        print("Log scale detected and time[0] is 0. Changing to 1 for plotting.")
        times[0] = 1

    # We use ax.plot here because we have manually created the step-like coordinates.
    ax.plot(times, sizes, label=pop_name)

    ax.set_xlabel("Time (generations ago)")
    ax.set_ylabel("Effective Population Size (Ne)")
    
    if log_time:
        ax.set_xscale("log")
    
    # Explicitly set y-axis limits to include all data, with some padding.
    if len(sizes) > 0:
        min_size = np.min(sizes)
        max_size = np.max(sizes)
        y_pad = (max_size - min_size) * 0.05
        # Ensure some padding even if min and max are the same
        if y_pad == 0:
            y_pad = max_size * 0.1
        final_y_top = max_size + y_pad
        ax.set_ylim(bottom=0, top=final_y_top)
        print(f"Manual Y-axis limits set: bottom=0, top={final_y_top}")

    ax.set_title("Demographic History")
    ax.legend()
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    
    # Save the plot
    filename = "demographic_history.png"
    plt.savefig(filename)
    print(f"Demographic history plot saved to {filename}")
    print("--- Plotting Complete ---")
    plt.close()


if __name__ == "__main__":
    # 1. Choose a demographic model to test
    # You can comment/uncomment different models to test them.
    # print("Testing single_bottleneck model...")
    # demo_model = single_bottleneck(N0=10000, N_bot=1000, t_bot_start=2000, t_bot_end=1000)
    # demo_model = constant_size(N0=5000)
    # demo_model = single_expansion(N0=10000, N_exp=20000, t_exp=1500)
    # demo_model = exponential_growth(N0=10000, t_growth_start=500, rate=0.01)

    print("Testing Gaussian Process model with a sharp bottleneck...")
    # Generate a random history using a GP, and add a bottleneck
    # Using a fixed random_state for reproducibility during testing
    gp_events, N0 = generate_gp_history(
        random_state=42, 
        add_sharp_bottleneck=True,
        bottleneck_strength_reduction=0.05 # Reduce to 5% of size
    )
    demo_model = piecewise_constant(events=gp_events, N0=N0)

    # 2. Plot the demographic history to verify it's correct
    plot_demography(demo_model, log_time=True) # GP looks better on log-log plot

    # 3. Run a small simulation with this model
    print("Running a small simulation...")
    ts = msprime.sim_ancestry(
        samples=20,
        demography=demo_model,
        sequence_length=1_000_000,  # 1Mb
        recombination_rate=1e-8,
        random_seed=42
    )
    mts = msprime.sim_mutations(ts, rate=1e-8, random_seed=42)
    print("Simulation finished.")
    print(f"  - Number of samples: {mts.num_samples}")
    print(f"  - Number of trees: {mts.num_trees}")
    print(f"  - Number of mutations (variants): {mts.num_mutations}")
    print(f"  - Total sequence length: {mts.sequence_length} bp") 