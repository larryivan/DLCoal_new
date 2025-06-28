import msprime
import numpy as np

def constant_size(N0=1e4):
    """
    Creates a demographic model for a population with a constant size.

    Args:
        N0 (float): The effective population size. Defaults to 10,000.

    Returns:
        msprime.Demography: A msprime Demography object.
    """
    demography = msprime.Demography()
    demography.add_population(name="pop", initial_size=N0)
    return demography

def single_bottleneck(N0=1e4, N_bot=1e3, t_bot_start=1000, t_bot_end=1200):
    """
    Creates a demographic model for a single bottleneck event.

    The model is:
    - Constant size N0 until t_bot_end.
    - Population size is N_bot between t_bot_end and t_bot_start.
    - Population size is N0 before t_bot_start.

    Note: Time is in generations ago.

    Args:
        N0 (float): The initial effective population size.
        N_bot (float): The effective population size during the bottleneck.
        t_bot_start (float): The time (generations ago) when the bottleneck started.
        t_bot_end (float): The time (generations ago) when the bottleneck ended.

    Returns:
        msprime.Demography: A msprime Demography object.
    """
    demography = msprime.Demography()
    demography.add_population(name="pop", initial_size=N0)
    demography.add_population_parameters_change(time=t_bot_end, initial_size=N_bot, population="pop")
    demography.add_population_parameters_change(time=t_bot_start, initial_size=N0, population="pop")
    return demography

def single_expansion(N0=1e4, N_exp=2e4, t_exp=1000):
    """
    Creates a demographic model for a single, instantaneous population expansion.

    The model is:
    - Constant size N_exp from the present until t_exp.
    - Constant size N0 before t_exp.

    Note: Time is in generations ago.

    Args:
        N0 (float): The ancestral effective population size.
        N_exp (float): The effective population size after expansion.
        t_exp (float): The time (generations ago) of the expansion event.

    Returns:
        msprime.Demography: A msprime Demography object.
    """
    demography = msprime.Demography()
    demography.add_population(name="pop", initial_size=N_exp)
    demography.add_population_parameters_change(time=t_exp, initial_size=N0, population="pop")
    return demography

def exponential_growth(N0=1e4, t_growth_start=1000, rate=0.005):
    """
    Creates a demographic model for a recent, continuous exponential growth.

    The model is:
    - Exponential growth starting at t_growth_start (generations ago) until the present.
    - Constant size N0 before t_growth_start.
    
    A positive rate indicates growth, a negative rate indicates decline.

    Note: Time is in generations ago.

    Args:
        N0 (float): The ancestral effective population size.
        t_growth_start (float): The time (generations ago) when growth began.
        rate (float): The exponential growth rate. 

    Returns:
        msprime.Demography: A msprime Demography object.
    """
    demography = msprime.Demography()
    demography.add_population(
        name="pop", 
        initial_size=N0, 
        growth_rate=rate
    )
    # At t_growth_start, the growth stops, and size becomes N0
    demography.add_population_parameters_change(time=t_growth_start, initial_size=N0, growth_rate=0, population="pop")
    return demography

def piecewise_constant(events, N0=1e4):
    """
    Creates a demographic model with piecewise constant population sizes.

    Args:
        events (list): A list of (time, size) tuples specifying the population
                       size changes. The list must be sorted by time.
                       Example: [(100, 5000), (500, 20000)]
        N0 (float): The population size at time 0 (present).

    Returns:
        msprime.Demography: A msprime Demography object.
    """
    demography = msprime.Demography()
    demography.add_population(name="pop", initial_size=N0)

    # The events list should be sorted by time
    sorted_events = sorted(events, key=lambda x: x[0])

    for time, size in sorted_events:
        demography.add_population_parameters_change(time=time, initial_size=size, population="pop")

    return demography 