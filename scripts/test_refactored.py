#!/usr/bin/env python3
"""
Test script to verify the refactored code works correctly.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported correctly."""
    try:
        from iblpt.agents.ibl import IBLAgent
        from iblpt.agents.pt_ibl import PTIBLAgent
        from iblpt.data.loader import load_dataset
        from iblpt.human.metrics import human_r_ts_est
        from iblpt.metrics import msd, corr, aic_from_msd
        from iblpt.optimize import fit_ibl, fit_pt
        from iblpt.plotting import plot_r_rate_curves
        from iblpt.simulate import eval_ts
        from iblpt.config import PARAMETER_BOUNDS, DEFAULT_PARAMETERS
        
        print("✓ All modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_agent_creation():
    """Test that agents can be created with parameters."""
    try:
        from iblpt.agents.ibl import IBLAgent
        from iblpt.agents.pt_ibl import PTIBLAgent
        
        # Test IBL agent
        ibl_agent = IBLAgent(d=5.27, s=1.46, p=0.09)
        print("✓ IBL agent created successfully")
        
        # Test PT+IBL agent
        pt_agent = PTIBLAgent(d=5.27, s=1.46, p=0.09, α=0.8, β=0.8, lam=2.25)
        print("✓ PT+IBL agent created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent creation error: {e}")
        return False

def test_data_loading():
    """Test that data can be loaded."""
    try:
        from iblpt.data.loader import load_dataset
        from iblpt.config import DATA_PATHS
        
        # Test loading estimation dataset
        est_dataset = load_dataset(DATA_PATHS['estimation'])
        print(f"✓ Estimation dataset loaded: {len(est_dataset)} problems")
        
        # Test loading competition dataset
        comp_dataset = load_dataset(DATA_PATHS['competition'])
        print(f"✓ Competition dataset loaded: {len(comp_dataset)} problems")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False

def test_metrics():
    """Test that metrics functions work."""
    try:
        import numpy as np
        from iblpt.metrics import msd, corr, aic_from_msd
        
        # Test data
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        # Test MSD
        msd_val = msd(x, y)
        print(f"✓ MSD calculation: {msd_val:.6f}")
        
        # Test correlation
        corr_val = corr(x, y)
        print(f"✓ Correlation calculation: {corr_val:.6f}")
        
        # Test AIC
        aic_val = aic_from_msd(msd_val, 3)
        print(f"✓ AIC calculation: {aic_val:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Metrics error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing refactored IBL/PT+IBL code...\n")
    
    tests = [
        test_imports,
        test_agent_creation,
        test_data_loading,
        test_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The refactored code is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

