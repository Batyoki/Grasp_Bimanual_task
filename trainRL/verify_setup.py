#!/usr/bin/env python3
"""
Verification script to check that all components are properly installed.
Run this after setup to verify everything is working.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version >= 3.10"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}", end=" ")
    if version >= (3, 10):
        print("✓")
        return True
    else:
        print("✗ (Need 3.10+)")
        return False

def check_imports():
    """Check critical imports"""
    packages = [
        ("torch", "PyTorch"),
        ("gymnasium", "Gymnasium"),
        ("omegaconf", "OmegaConf"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("isaaclab", "IsaacLab"),
    ]
    
    all_ok = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"{name:25} ✓")
        except ImportError:
            print(f"{name:25} ✗ (not installed)")
            all_ok = False
    
    return all_ok

def check_directories():
    """Check directory structure"""
    required_dirs = [
        "bimanual_vx300s_env",
        "bimanual_vx300s_env/mdp",
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"Directory {dir_path:30} ✓")
        else:
            print(f"Directory {dir_path:30} ✗ (missing)")
            all_ok = False
    
    return all_ok

def check_files():
    """Check key files"""
    required_files = [
        "train_ppo.py",
        "evaluate.py",
        "quickstart.py",
        "utils.py",
        "requirements.txt",
        "config.yaml",
        "README.md",
        "SETUP_GUIDE.md",
        "MIGRATION_GUIDE.md",
        "bimanual_vx300s_env/__init__.py",
        "bimanual_vx300s_env/vx300s_env_cfg.py",
        "bimanual_vx300s_env/mdp/__init__.py",
        "bimanual_vx300s_env/mdp/actions.py",
        "bimanual_vx300s_env/mdp/observations.py",
        "bimanual_vx300s_env/mdp/rewards.py",
        "bimanual_vx300s_env/mdp/terminations.py",
    ]
    
    all_ok = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"File {file_path:45} ✓")
        else:
            print(f"File {file_path:45} ✗ (missing)")
            all_ok = False
    
    return all_ok

def check_environment_import():
    """Test environment import"""
    try:
        from bimanual_vx300s_env import BimanualVX300sEnvCfg
        print("Environment import test         ✓")
        return True
    except Exception as e:
        print(f"Environment import test         ✗ ({e})")
        return False

def main():
    """Run all checks"""
    print("\n" + "="*70)
    print("  Bimanual VX300s IsaacLab Environment - Verification")
    print("="*70 + "\n")
    
    print("1. Checking Python Version:")
    print("-" * 70)
    py_ok = check_python_version()
    print()
    
    print("2. Checking Python Packages:")
    print("-" * 70)
    imports_ok = check_imports()
    print()
    
    print("3. Checking Directory Structure:")
    print("-" * 70)
    dirs_ok = check_directories()
    print()
    
    print("4. Checking Files:")
    print("-" * 70)
    files_ok = check_files()
    print()
    
    print("5. Checking Environment Import:")
    print("-" * 70)
    env_ok = check_environment_import()
    print()
    
    print("="*70)
    if py_ok and imports_ok and dirs_ok and files_ok and env_ok:
        print("✓ ALL CHECKS PASSED - Environment is ready!")
        print("\nNext steps:")
        print("  1. Read README.md for usage guide")
        print("  2. Run: python quickstart.py")
        print("  3. Or directly train: python train_ppo.py --num-envs 4")
        print("="*70 + "\n")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Please fix the issues above")
        print("\nTroubleshooting:")
        if not imports_ok:
            print("  - Install missing packages: pip install -r requirements.txt")
        if not dirs_ok or not files_ok:
            print("  - Check directory structure in README.md")
        print("="*70 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
