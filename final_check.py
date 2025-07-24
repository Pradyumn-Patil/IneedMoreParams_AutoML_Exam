#!/usr/bin/env python3
"""
Final System Check - Verifies the extended AutoML system is ready for deployment

This script performs a quick check of all major components to ensure
the system is properly installed and configured.
"""

import sys
import importlib
from pathlib import Path
import subprocess

def check_module(module_name, component_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {component_name}")
        return True
    except ImportError as e:
        print(f"❌ {component_name}: {str(e)}")
        return False

def check_dependencies():
    """Check all required dependencies."""
    print("\n🔍 Checking Dependencies...")
    print("-" * 40)
    
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("sklearn", "Scikit-learn"),
        ("optuna", "Optuna"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("sentence_transformers", "Sentence-Transformers"),
        ("gensim", "Gensim"),
        ("nltk", "NLTK"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
    ]
    
    all_ok = True
    for module, name in dependencies:
        if not check_module(module, name):
            all_ok = False
            
    return all_ok

def check_automl_modules():
    """Check all AutoML modules."""
    print("\n🔧 Checking AutoML Modules...")
    print("-" * 40)
    
    modules = [
        ("automl.core", "Core AutoML"),
        ("automl.datasets", "Dataset Loaders"),
        ("automl.models", "Neural Models"),
        ("automl.preprocessing", "Text Preprocessing"),
        ("automl.embeddings", "Embeddings Module"),
        ("automl.traditional_ml", "Traditional ML Models"),
        ("automl.transformers_advanced", "Advanced Transformers"),
        ("automl.models_extended", "Extended Neural Models"),
        ("automl.augmentation_advanced", "Advanced Augmentation"),
        ("automl.fidelity_manager", "Fidelity Management"),
        ("automl.ensemble", "Ensemble Methods"),
        ("automl.meta_learning", "Meta-Learning"),
    ]
    
    all_ok = True
    for module, name in modules:
        if not check_module(module, name):
            all_ok = False
            
    return all_ok

def check_scripts():
    """Check if main scripts exist."""
    print("\n📜 Checking Scripts...")
    print("-" * 40)
    
    scripts = [
        "run.py",
        "run_automl_complete.py",
        "run_extended_automl.py",
        "validate_system.py",
        "run_all_experiments.py",
    ]
    
    all_ok = True
    for script in scripts:
        if Path(script).exists():
            print(f"✅ {script}")
        else:
            print(f"❌ {script} not found")
            all_ok = False
            
    return all_ok

def check_gpu():
    """Check GPU availability."""
    print("\n🖥️  Checking Hardware...")
    print("-" * 40)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  No GPU available - will use CPU")
    except:
        print("⚠️  Could not check GPU status")
        
def run_quick_test():
    """Run a very quick functionality test."""
    print("\n🧪 Running Quick Test...")
    print("-" * 40)
    
    try:
        # Test basic import and instantiation
        from automl.core import TextAutoML
        from automl.embeddings import EmbeddingManager
        from automl.traditional_ml import TraditionalMLFactory
        
        # Test creating instances
        automl = TextAutoML(approach='logistic', seed=42)
        print("✅ Core AutoML instantiation")
        
        manager = EmbeddingManager()
        print("✅ Embedding manager instantiation")
        
        model = TraditionalMLFactory.create_model('svm_linear')
        print("✅ Traditional ML factory")
        
        print("\n✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def main():
    """Run all checks."""
    print("=" * 60)
    print("🚀 Extended AutoML System - Final Check")
    print("=" * 60)
    
    # Run all checks
    deps_ok = check_dependencies()
    modules_ok = check_automl_modules()
    scripts_ok = check_scripts()
    check_gpu()
    test_ok = run_quick_test()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    all_ok = deps_ok and modules_ok and scripts_ok and test_ok
    
    if all_ok:
        print("✅ All checks passed! System is ready for deployment.")
        print("\nNext steps:")
        print("1. Run validation: python validate_system.py")
        print("2. Test on a dataset: python run_extended_automl.py --dataset amazon --approach auto")
        print("3. Run full experiments for submission")
    else:
        print("❌ Some checks failed. Please address the issues above.")
        print("\nCommon fixes:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Ensure all files were properly created")
        print("3. Check Python version (3.8+ recommended)")
        
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())