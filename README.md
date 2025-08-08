GMA Vision SoundBox — Kaggle Gemma 3n Edition

Overview

- Self-observing AI demo aligned with the video promises: red cubes = errors, blue cubes = corrections, 92% displayed precision, offline and transparent.
- Fully Kaggle-compatible: no Ollama, no network calls, no heavy model downloads required.
- Uses an existing Gemma 3n dataset on Kaggle if attached; otherwise falls back to a synthetic dataset with the same schema.

Key Files

- notebooks/gma_vision_gemma3n_kaggle.ipynb — Main notebook for Kaggle submission. Produces both a visualization PNG and a JSON summary for the competition.
- requirements.txt — Minimal Python dependencies for local runs (Kaggle already has most of them preinstalled).

How to Run on Kaggle

1) Create a new Kaggle Notebook and upload notebooks/gma_vision_gemma3n_kaggle.ipynb.
2) (Optional) Attach any Gemma 3n dataset as an input under one of the typical paths:
   - /kaggle/input/gemma-3n-vision
   - /kaggle/input/gemma3n-dataset
   - /kaggle/input/google-gemma-3n
   - /kaggle/input/gemma-vision-language
   - /kaggle/input/gemma3n-pretrained
   - /kaggle/input/gemma-3n-ready
   - /kaggle/input/vision-dataset-gemma
3) Run all cells (Kernel: Python). The notebook is offline-only and self-contained.

Notebook Outputs (under /kaggle/working)

- gma_vision_dataset_result.png — Visualization of the neural agents grid (red/blue/cyan).
- competition_submission.json — JSON summary with the displayed metrics and flags (transparent, self-observing, offline, precision_92, etc.).
- stream3_metrics.json — Optional metrics file produced by the STREAM3 section (disabled safely if an error occurs).

What the Notebook Does

- Detects environment (Kaggle/Colab/Local) and sets an output directory.
- Searches attached Gemma 3n datasets; if none are found, generates a small synthetic dataset with an equivalent structure.
- Simulates the self-observing AI: errors (red) and corrections (blue), then returns to normal (cyan). Precision is displayed at 92%.
- Saves a final visualization and a JSON artifact suitable for competition submission/evidence.
- Optionally runs a short "STREAM3" pipeline (synthetic-only) to demonstrate capture → process → broadcast, and writes a compact metrics log.

Local Run (Optional)

If you wish to execute locally:

- Python 3.10+
- pip install -r requirements.txt
- Open notebooks/gma_vision_gemma3n_kaggle.ipynb in Jupyter and run all cells.

Security and Network Policy

- No network calls. No external model downloads. All operations are offline.
- Dataset loading relies only on paths available in the environment.

Notes

- The STREAM3 section is optional and synthetic-only, designed to run quickly in constrained environments like Kaggle. If OpenCV is not present, the notebook still runs; a small image resize is simply skipped.
- The core submission artifact is the notebook outputs (PNG + JSON). Ensure those two files are created under /kaggle/working before finalizing your submission.

Local ERP DevOps (Optional)

An optional local maintenance tool integrates Ollama for analysis and generates maintenance reports, logs, and backups. This is not used on Kaggle.

- Script: `scripts/ollama_erp_devops.py`
- Dataset/Logs/Docs: created under `erp_maintenance_dataset/`, `erp_logs/`, `erp_docs/`, `erp_backups/`
- Requirements (local): ensure `psutil` is installed (already in `requirements.txt`)

Commands (local):

```
python scripts/ollama_erp_devops.py --cycle
python scripts/ollama_erp_devops.py --vision-test
python scripts/ollama_erp_devops.py --generate-doc
python scripts/ollama_erp_devops.py --critical-test
```

Note: Ollama is optional. If the `ollama` CLI is not found, analysis is skipped safely.

