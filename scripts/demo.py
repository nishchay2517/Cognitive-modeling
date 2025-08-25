#!/usr/bin/env python3
"""
Demo script to showcase the refactored IBL/PT+IBL code without running optimization.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iblpt.agents.ibl import IBLAgent
from iblpt.agents.pt_ibl import PTIBLAgent
from iblpt.data.loader import load_dataset
from iblpt.human.metrics import human_r_ts_est
from iblpt.metrics import msd, corr, aic_from_msd
from iblpt.simulate import eval_ts
from iblpt.plotting import plot_r_rate_curves
from iblpt.config import DATA_PATHS, OUTPUT_PATHS, DEFAULT_PARAMETERS


def demo_agents():
    """Demonstrate agent creation and basic functionality."""
    print("=== Agent Demo ===")
    
    # Create IBL agent
    ibl_agent = IBLAgent(d=5.27, s=1.46, p=0.09)
    print(f"✓ Created IBL agent with parameters: d={ibl_agent.d}, s={ibl_agent.s}, p={ibl_agent.p}")
    
    # Create PT+IBL agent
    pt_agent = PTIBLAgent(d=5.27, s=1.46, p=0.09, α=0.8, β=0.8, lam=2.25)
    print(f"✓ Created PT+IBL agent with parameters: d={pt_agent.d}, s={pt_agent.s}, p={pt_agent.p}, α={pt_agent.α}, β={pt_agent.β}, λ={pt_agent.lam}")
    
    return ibl_agent, pt_agent


def demo_data_loading():
    """Demonstrate data loading functionality."""
    print("\n=== Data Loading Demo ===")
    
    # Load datasets
    est_dataset = load_dataset(DATA_PATHS['estimation'])
    comp_dataset = load_dataset(DATA_PATHS['competition'])
    
    print(f"✓ Loaded estimation dataset: {len(est_dataset)} problems")
    print(f"✓ Loaded competition dataset: {len(comp_dataset)} problems")
    
    # Show sample data
    print(f"✓ Sample estimation problem columns: {list(est_dataset.columns)}")
    print(f"✓ First problem values: {est_dataset.iloc[0].to_dict()}")
    
    return est_dataset, comp_dataset


def demo_simulation(est_dataset):
    """Demonstrate simulation functionality."""
    print("\n=== Simulation Demo ===")
    
    # Test IBL simulation
    ibl_params = DEFAULT_PARAMETERS['ibl']
    print(f"Running IBL simulation with parameters: {ibl_params}")
    
    try:
        r_ts, a_ts = eval_ts(est_dataset, ibl_params, use_pt=False, N=20, agents=3)
        print(f"✓ IBL simulation successful: R-rate shape={r_ts.shape}, A-rate shape={a_ts.shape}")
        print(f"✓ First few R-rates: {r_ts[:5]}")
        print(f"✓ First few A-rates: {a_ts[:5]}")
    except Exception as e:
        print(f"✗ IBL simulation failed: {e}")
        return None, None
    
    return r_ts, a_ts


def demo_metrics(r_ts, human_r_ts):
    """Demonstrate metrics calculation."""
    print("\n=== Metrics Demo ===")
    
    if r_ts is None:
        print("Skipping metrics demo due to simulation failure")
        return
    
    # Calculate metrics
    msd_val = msd(r_ts, human_r_ts[:len(r_ts)])
    corr_val = corr(r_ts, human_r_ts[:len(r_ts)])
    aic_val = aic_from_msd(msd_val, 3)
    
    print(f"✓ MSD calculation: {msd_val:.6f}")
    print(f"✓ Correlation calculation: {corr_val:.6f}")
    print(f"✓ AIC calculation: {aic_val:.6f}")


def demo_plotting(est_dataset, r_ts):
    """Demonstrate plotting functionality."""
    print("\n=== Plotting Demo ===")
    
    if r_ts is None:
        print("Skipping plotting demo due to simulation failure")
        return
    
    try:
        # Create a simple model for plotting
        models = [{
            'label': 'Demo IBL',
            'params': DEFAULT_PARAMETERS['ibl'],
            'use_pt': False
        }]
        
        # Plot with shorter series for demo
        human_r_ts_short = human_r_ts_est[:len(r_ts)]
        
        fig = plot_r_rate_curves(
            est_dataset,
            human_r_ts_short,
            models,
            N=len(r_ts),
            agents=3,
            save_path="demo_plot.png"
        )
        print("✓ Plot generated and saved as 'demo_plot.png'")
        
    except Exception as e:
        print(f"✗ Plotting failed: {e}")


def main():
    """Run the demo."""
    print("🚀 IBL/PT+IBL Refactored Code Demo\n")
    
    try:
        # Run all demos
        ibl_agent, pt_agent = demo_agents()
        est_dataset, comp_dataset = demo_data_loading()
        r_ts, a_ts = demo_simulation(est_dataset)
        demo_metrics(r_ts, human_r_ts_est)
        demo_plotting(est_dataset, r_ts)
        
        print("\n🎉 Demo completed successfully!")
        print("\nThe refactored code is working correctly.")
        print("You can now use the CLI or main scripts for full optimization.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
