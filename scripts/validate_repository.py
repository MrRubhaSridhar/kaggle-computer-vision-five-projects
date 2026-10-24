#!/usr/bin/env python3
"""
scripts/validate_repository.py
Performs automated verification of all code files, AST syntax parsing,
dataset directories, metrics JSON files, and visual results.
"""

import os
import sys
import glob
import json
import py_compile

def check_syntax():
    print("--- 1. Python Syntax Validation ---")
    py_files = sorted(glob.glob("**/*.py", recursive=True))
    py_files = [f for f in py_files if not f.startswith(".venv") and not f.startswith("venv")]
    syntax_ok = True
    for pf in py_files:
        try:
            py_compile.compile(pf, doraise=True)
            print(f"[✓] Syntax valid: {pf}")
        except py_compile.PyCompileError as e:
            print(f"[✗] Syntax ERROR in {pf}: {e}")
            syntax_ok = False
    return syntax_ok

def check_artifacts():
    print("\n--- 2. Project Artifacts & Metrics Verification ---")
    projects = [
        ("01_lane_detection", ["results/evaluation_metrics.json", "results/summary_plots/lane_detection_comparison.png", "results/sample_clips/annotated_lane_departure_demo.mp4"]),
        ("02_fruit_quality_inspection", ["results/metrics.json", "results/confusion_matrix.png", "results/class_distribution.png", "results/training_curves.png", "results/predictions/correct_samples.png"]),
        ("03_traffic_sign_recognition", ["results/metrics.json", "results/confusion_matrix.png", "results/class_distribution.png", "results/training_curves.png", "results/predictions/sample_predictions.png"]),
        ("04_robotic_object_detection", ["results/metrics.json", "results/dataset_exploration.png", "results/sample_clips/detected_hardhat.mp4"]),
        ("05_explainable_ai", ["results/metrics.json", "results/confusion_matrix.png", "results/training_curves.png", "results/dataset_samples.png", "results/gradcam/gradcam_comparison_grid.png"]),
    ]
    all_ok = True
    for proj_dir, expected_files in projects:
        print(f"\nVerifying {proj_dir}...")
        for rel_p in expected_files:
            full_p = os.path.join(proj_dir, rel_p)
            if os.path.exists(full_p):
                size_kb = os.path.getsize(full_p) / 1024
                print(f"  [✓] Found {rel_p} ({size_kb:.1f} KB)")
                # If json, verify validity
                if full_p.endswith(".json"):
                    with open(full_p) as jf:
                        d = json.load(jf)
                        print(f"      (Valid JSON with keys: {list(d.keys())[:4]}...)")
            else:
                print(f"  [✗] MISSING: {full_p}")
                all_ok = False
    return all_ok

def main():
    print("=" * 60)
    print("       Running Repository Integrity Verification       ")
    print("=" * 60)
    syntax_pass = check_syntax()
    artifacts_pass = check_artifacts()
    print("\n" + "=" * 60)
    if syntax_pass and artifacts_pass:
        print("ALL VERIFICATION CHECKS PASSED: SUCCESS")
        print("=" * 60)
        sys.exit(0)
    else:
        print("SOME VERIFICATION CHECKS FAILED")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
