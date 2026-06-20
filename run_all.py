"""Reproduce the whole pipeline end to end. Run from the repository root:  python run_all.py"""
import subprocess, sys, os
S = os.path.join(os.path.dirname(__file__), "src")
for step in ["generate_dataset.py", "train_surrogate.py", "optimise.py", "sensitivity.py", "build_datasheet.py"]:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, os.path.join(S, step)], check=True)
print("\nDone. See data/, models/, results/.")
