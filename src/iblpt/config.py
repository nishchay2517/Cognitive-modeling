"""
Configuration module for IBL and PT+IBL models.
"""

# Optimization parameters
OPTIMIZATION_CONFIG = {
    'maxiter': 15,
    'popsize': 8,
    'updating': 'deferred',
    'workers': -1,
    'tol': 1e-2
}

# Model parameter bounds
PARAMETER_BOUNDS = {
    'ibl': [(0.01, 10), (0.01, 10), (0.0, 1.0)],  # decay, noise, inertia
    'pt': [(0.01, 10), (0.01, 10), (0.0, 1.0),     # decay, noise, inertia
           (0.2, 1.0), (0.2, 1.0), (0.1, 5.0)]     # alpha, beta, lambda
}

# Default parameters
DEFAULT_PARAMETERS = {
    'ibl': (5.27, 1.46, 0.09),
    'pt': None  # Will be optimized
}

# Pre-optimized parameters (from previous runs)
PRE_OPTIMIZED_PARAMETERS = {
    'ibl': [4.59856917, 0.04554824, 0.01635943]
}

# Simulation parameters
SIMULATION_CONFIG = {
    'N': 100,      # number of trials
    'agents': 5,   # Monte Carlo replications per problem
    'plot_agents': 20  # agents for plotting
}

# Data paths
DATA_PATHS = {
    'estimation': 'repeated/data/60estimationset.dat',
    'competition': 'repeated/data/60competitionset.dat'
}

# Output paths
OUTPUT_PATHS = {
    'plot': 'IBLPT_vs_human.png'
}



