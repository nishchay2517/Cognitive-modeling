#!/usr/bin/env python3
"""
Command-line interface for IBL and PT+IBL model optimization.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iblpt.data.loader import load_dataset
from iblpt.optimize import fit_ibl, fit_pt, create_callback
from iblpt.metrics import summarize
from iblpt.plotting import plot_r_rate_curves
from iblpt.human.metrics import human_r_ts_est, human_a_ts_est
from iblpt.config import (
    PARAMETER_BOUNDS, DEFAULT_PARAMETERS,
    SIMULATION_CONFIG, DATA_PATHS, OUTPUT_PATHS
)


def run_optimization(model_type, dataset_path=None, output_dir=None, workers=1):
    """Run optimization for specified model type."""
    print(f"Running optimization for {model_type} model...")
    print(f"Using {workers} worker(s) for optimization")
    
    # Load datasets
    if dataset_path:
        est_dataset = load_dataset(dataset_path)
    else:
        est_dataset = load_dataset(DATA_PATHS['estimation'])
    
    comp_dataset = load_dataset(DATA_PATHS['competition'])
    
    if model_type == "ibl":
        callback = create_callback("IBL")
        res = fit_ibl(
            PARAMETER_BOUNDS['ibl'], 
            est_dataset, 
            callback=callback,
            workers=workers
        )
        params = tuple(res.x)
        use_pt = False
        label = "Optimized IBL"
        
    elif model_type == "pt":
        callback = create_callback("PT")
        res = fit_pt(
            PARAMETER_BOUNDS['pt'], 
            est_dataset, 
            callback=callback,
            workers=workers
        )
        params = tuple(res.x)
        use_pt = True
        label = "PT-Optimized IBL"
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"\nOptimization complete! Best parameters: {params}")
    
    # Generate summary
    summarize(label, params, use_pt, est_dataset, comp_dataset)
    
    # Generate plot
    if output_dir:
        plot_path = Path(output_dir) / OUTPUT_PATHS['plot']
    else:
        plot_path = OUTPUT_PATHS['plot']
    
    plot_r_rate_curves(
        est_dataset,
        human_r_ts_est,
        models=[{'label': label, 'params': params, 'use_pt': use_pt}],
        N=SIMULATION_CONFIG['N'],
        agents=SIMULATION_CONFIG['plot_agents'],
        save_path=str(plot_path)
    )
    
    return params


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="IBL and PT+IBL Model Optimization")
    parser.add_argument(
        "model", 
        choices=["ibl", "pt"], 
        help="Model type to optimize"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        help="Path to estimation dataset"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Output directory for plots"
    )
    parser.add_argument(
        "--compare", 
        action="store_true", 
        help="Compare with default parameters"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1, 
        help="Number of workers for optimization (default: 1, use -1 for all cores)"
    )
    
    args = parser.parse_args()
    
    try:
        # Run optimization
        params = run_optimization(args.model, args.dataset, args.output_dir, args.workers)
        
        # Compare with default if requested
        if args.compare and args.model == "ibl":
            print("\n=== COMPARISON WITH DEFAULT ===")
            est_dataset = load_dataset(DATA_PATHS['estimation'])
            comp_dataset = load_dataset(DATA_PATHS['competition'])
            summarize("Default IBL", DEFAULT_PARAMETERS['ibl'], False, est_dataset, comp_dataset)
            summarize("Optimized IBL", params, False, est_dataset, comp_dataset)
            
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
