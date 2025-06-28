import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def generate_gp_history(
    num_control_points=5,
    max_time=100_000,
    min_ne=100,
    max_ne=100_000,
    num_discretization_points=100,
    add_sharp_bottleneck=False,
    bottleneck_strength_reduction=0.01, # e.g., reduce to 1% of original size
    random_state=None
):
    """
    Generates a smooth, random demographic history using a Gaussian Process,
    with an option to add a sharp, sudden bottleneck event.

    The function operates in log-space for both time and population size
    to generate more natural-looking curves.

    Args:
        num_control_points (int): The number of random points to anchor the GP.
        max_time (float): The maximum time (in generations ago) for the history.
        min_ne (float): The minimum possible effective population size.
        max_ne (float): The maximum possible effective population size.
        num_discretization_points (int): The number of points to use for the
                                         final piecewise-constant approximation.
        add_sharp_bottleneck (bool): If True, add a sharp bottleneck to the history.
        bottleneck_strength_reduction (float): The factor by which to reduce Ne
                                               during the bottleneck (e.g., 0.01 for 1%).
        random_state (int, optional): A seed for the random number generator.

    Returns:
        tuple: A tuple containing:
            - list: A list of (time, size) events for msprime.
            - float: The population size at time 0 (N0).
    """
    if random_state is not None:
        np.random.seed(random_state)

    # 1. Define random control points in log-space for time and Ne
    # We use log-space for time to better capture ancient events.
    # Time points are chosen from log(1) up to log(max_time).
    # We add a small epsilon to time to avoid log(0).
    control_times = np.exp(np.random.uniform(
        np.log(10), np.log(max_time), size=(num_control_points, 1)
    ))
    control_log_ne = np.random.uniform(
        np.log(min_ne), np.log(max_ne), size=num_control_points
    )

    # 2. Fit a Gaussian Process Regressor
    # A length scale of ~0.5 on log-time seems to produce reasonable curves.
    kernel = 1.0 * RBF(length_scale=0.5, length_scale_bounds=(1e-1, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
    gp.fit(np.log(control_times), control_log_ne)

    # 3. Predict a smooth curve over a dense grid of time points
    log_time_grid = np.linspace(np.log(1), np.log(max_time), num_discretization_points)
    predicted_log_ne, sigma = gp.predict(log_time_grid[:, np.newaxis], return_std=True)

    # 4. Convert back from log-space and create piecewise constant events
    event_times = np.exp(log_time_grid)
    event_sizes = np.exp(predicted_log_ne)

    # Ensure Ne values are within bounds
    event_sizes = np.clip(event_sizes, min_ne, max_ne)

    # N0 is the size at the first time point (closest to the present)
    N0 = event_sizes[0]
    
    # Create the event list for msprime, skipping the first element (which is N0)
    events = []
    for i in range(1, len(event_times)):
        # msprime events are (time, size)
        events.append((event_times[i], event_sizes[i]))

    # 5. Optionally add a sharp bottleneck event
    if add_sharp_bottleneck:
        # Pick a random time for the bottleneck to start, not too close to the present
        bottleneck_start_time = np.random.uniform(500, max_time / 2)
        # Make the bottleneck very short
        bottleneck_end_time = bottleneck_start_time + 2

        # Find the size before the bottleneck
        original_size_at_bottleneck = N0
        # Find the Ne value right before the bottleneck starts
        for t, s in reversed(events):
            if t <= bottleneck_start_time:
                original_size_at_bottleneck = s
                break
        
        bottleneck_size = max(min_ne, original_size_at_bottleneck * bottleneck_strength_reduction)

        print(f"Adding sharp bottleneck: start={bottleneck_start_time:.2f}, "
              f"end={bottleneck_end_time:.2f}, size={bottleneck_size:.2f}")

        # Insert bottleneck events and keep the list sorted by time
        events.append((bottleneck_start_time, bottleneck_size))
        events.append((bottleneck_end_time, original_size_at_bottleneck))
        events.sort(key=lambda x: x[0])

    return events, N0 