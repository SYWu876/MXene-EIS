# MXene EIS Analysis (Nano Energy) — Reproducibility Package

This repository is a **ready-to-upload** GitHub/Zenodo structure to reproduce the MXene EIS fitting
and quantum-branch artifacts (QAOA/VQE–VQA) used in the manuscript.

## Structure
- `data/`    Raw and derived data (CSV/TXT): EIS, fit outputs, bounds/decoding, QUBO/Ising, heatmaps, robustness runs
- `src/`     Core Python modules (circuit model, decoding, NanoEnergy plotting, QAOA utilities)
- `scripts/` Reproducibility scripts (regenerate key CSVs and representative plots)
- `figures/` Pre-rendered manuscript figures (PNG 600 dpi + PDF)
- `tables/`  Pre-rendered manuscript tables (DOCX/PDF/CSV)

## Environment
```bash
conda env create -f environment.yml
conda activate mxene-eis-nanoenergy
```

## Reproduce key artifacts
```bash
python scripts/run_all.py
```
This regenerates:
- `data/MXene_EIS_fit_results_linearLS.csv`
- `figures/Regen_*` overlay plots (Figure 3-style)
- `figures/Regen_*` QAOA γ–β heatmaps (Figure S8-style)
- `data/Regen_FigureS9_*` best-shot tables + `figures/Regen_FigS9_*` top-10 counts

## Exact manuscript artifacts
The exact current manuscript artifacts are shipped in:
- `figures/Figure*.png/.pdf`, `figures/FigureS*.png/.pdf`
- `tables/Table*.docx/.pdf/.csv`, `tables/TableS*.docx/.pdf`

If you update raw data or bounds, rerun the scripts to regenerate updated artifacts.

## License
MIT License (edit year/owner as needed).
