#!/usr/bin/env python3
"""
scripts/check_environment.py
Validates the execution environment, GPU availability, dependencies, and Kaggle CLI access.
"""

import sys
import os
import shutil
import subprocess

def check_python():
    print(f"[+] Python Version: {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        print("    [!] Warning: Python >= 3.9 recommended.")
        return False
    return True

def check_hardware():
    print("\n--- Hardware & Storage ---")
    # Storage
    total, used, free = shutil.disk_usage(".")
    print(f"[+] Disk Space: Free={free / (1024**3):.2f} GB | Total={total / (1024**3):.2f} GB")
    
    # Memory
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem_avail = next(int(l.split()[1]) for l in lines if "MemAvailable:" in l) / 1024 / 1024
        print(f"[+] Available RAM: {mem_avail:.2f} GB")
    except Exception:
        print("[!] Could not query /proc/meminfo")

def check_gpu():
    print("\n--- GPU & Acceleration ---")
    try:
        import torch
        print(f"[+] PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"[+] CUDA Available: {cuda_avail}")
        if cuda_avail:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"    - Device {i}: {name} ({mem:.2f} GB VRAM, Compute Capability {cap[0]}.{cap[1]})")
        else:
            print("    [!] Warning: CUDA is not available. Execution will fall back to CPU.")
    except ImportError:
        print("[!] PyTorch is not installed.")

def check_dependencies():
    print("\n--- Core Libraries ---")
    packages = [
        ("cv2", "OpenCV"),
        ("torchvision", "TorchVision"),
        ("ultralytics", "Ultralytics YOLO"),
        ("sklearn", "Scikit-Learn"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
        ("seaborn", "Seaborn"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
        ("yaml", "PyYAML"),
    ]
    all_ok = True
    for mod_name, label in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"[+] {label}: {ver}")
        except ImportError:
            print(f"[!] {label} ({mod_name}) is MISSING!")
            all_ok = False
    return all_ok

def check_tools():
    print("\n--- External CLI Tools ---")
    tools = ["git", "kaggle", "gh"]
    for t in tools:
        path = shutil.which(t)
        if path:
            print(f"[+] {t}: {path}")
        else:
            print(f"[!] {t} is NOT found in PATH")
    
    # Kaggle Auth
    print("\n--- Kaggle Authentication Check ---")
    try:
        res = subprocess.run(["kaggle", "datasets", "list", "-p", "1"], capture_output=True, text=True, timeout=10)
        if res.returncode == 0:
            print("[+] Kaggle CLI Authentication: SUCCESS (token/credentials functional)")
        else:
            print(f"[!] Kaggle CLI returned code {res.returncode}. Output: {res.stderr.strip()[:100]}")
    except Exception as e:
        print(f"[!] Kaggle CLI invocation error: {e}")

def main():
    print("=" * 60)
    print("      Kaggle CV Five Projects - Environment Verification    ")
    print("=" * 60)
    check_python()
    check_hardware()
    check_gpu()
    deps_ok = check_dependencies()
    check_tools()
    print("=" * 60)
    if deps_ok:
        print("ENVIRONMENT STATUS: READY")
    else:
        print("ENVIRONMENT STATUS: MISSING DEPENDENCIES")
    print("=" * 60)

if __name__ == "__main__":
    main()
