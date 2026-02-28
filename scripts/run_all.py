"""Reproduce core artifacts for the MXene EIS + quantum analysis paper.

Usage:
  python scripts/run_all.py
"""
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parents[1]

def run(script_name: str):
    print(f"\n[RUN] {script_name}")
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / script_name)])

if __name__ == "__main__":
    run("01_fit_classical_nlls.py")
    run("02_plot_eis_overlays.py")
    run("03_qaoa_heatmaps.py")
    run("04_qaoa_bestshot_tables.py")
    print("\nDone. See /data for regenerated CSVs and /figures for regenerated plots (prefixed with Regen_).")
