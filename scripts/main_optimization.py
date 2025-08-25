#!/usr/bin/env python3
"""
Main optimization script for IBL and PT+IBL models.

This script demonstrates the refactored and organized code structure,
fitting models to human data and generating performance summaries.
"""

import numpy as np
from scipy.optimize import differential_evolution

# Import from organized modules
from iblpt.data.loader import load_dataset
from iblpt.optimize import fit_pt, create_callback
from iblpt.metrics import summarize
from iblpt.plotting import plot_r_rate_curves
from iblpt.human.metrics import human_r_ts_est
from iblpt.config import (
    PARAMETER_BOUNDS, DEFAULT_PARAMETERS,
    PRE_OPTIMIZED_PARAMETERS, SIMULATION_CONFIG, DATA_PATHS, OUTPUT_PATHS
)


def main():
    """Main execution function."""
    print("Loading datasets...")
    
    # Load datasets
    est_dataset = load_dataset(DATA_PATHS['estimation'])
    comp_dataset = load_dataset(DATA_PATHS['competition'])
    
    print(f"Loaded {len(est_dataset)} estimation problems and {len(comp_dataset)} competition problems")
    
    # Define parameter bounds
    bnds_ibl = PARAMETER_BOUNDS['ibl']
    bnds_pt = PARAMETER_BOUNDS['pt']
    
    # Pre-optimized IBL parameters (from previous runs)
    res_ibl = {'x': PRE_OPTIMIZED_PARAMETERS['ibl']}
    
    print('\nFitting for PT+IBL...')
    callback_pt = create_callback("PT")
    res_pt = differential_evolution(
        lambda x: fit_pt(bnds_pt, est_dataset, callback=callback_pt, workers=1).fun,
        bnds_pt,
        maxiter=15, popsize=8,
        updating='deferred', workers=1,  # Use single worker to avoid multiprocessing issues
        tol=1e-2, callback=callback_pt
    )
    
    # Default and optimized parameters
    default_ibl = DEFAULT_PARAMETERS['ibl']
    ibl_opt = tuple(res_ibl['x'])
    pt_opt = tuple(res_pt.x)
    
    print("\n=== FINAL RESULTS ===")
    
    # Generate summaries
    summarize("Default IBL", default_ibl, False, est_dataset, comp_dataset)
    summarize("Optimized IBL", ibl_opt, False, est_dataset, comp_dataset)
    summarize("PT-Optimized IBL", pt_opt, True, est_dataset, comp_dataset)
    
    # Plot results
    print("\nGenerating plots...")
    plot_r_rate_curves(
        est_dataset, 
        human_r_ts_est,
        models=[
            {'label': 'PT-Optimized IBL', 'params': pt_opt, 'use_pt': True}
        ],
        N=SIMULATION_CONFIG['N'],
        agents=SIMULATION_CONFIG['plot_agents'],
        save_path=OUTPUT_PATHS['plot']
    )
    
    print("\nOptimization complete!")


if __name__ == "__main__":
    main()
