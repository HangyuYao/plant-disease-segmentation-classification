# Plant Disease Segmentation and Classification

This project explores plant disease classification under three image-preparation settings:

- Original leaf images.
- Leaf-segmented images with the background removed.
- Lesion-only masked images that isolate diseased regions.

Version 1 of this repository is a portfolio-ready results dashboard. It summarizes prepared experiments and curated visuals; it does not perform live model inference and does not require trained model weights or the full datasets at runtime.

## Repository Contents

- `app/streamlit_app.py` - Streamlit dashboard for browsing experiment summaries, segmentation results, dataset inventory metadata, and curated figures.
- `reports/` - Small CSV summaries used by the dashboard.
- `assets/` - Lightweight figure exports and gallery metadata for dashboard display.
- `data/README.md` - Notes about expected local dataset archives and why large data is excluded.
- `docs/`, `scripts/`, `src/`, and `tests/` - Reproducibility notes, preparation utilities, reusable helpers, and project checks.

## Not Committed

Large research artifacts are intentionally excluded from the GitHub repository:

- Full datasets and extracted dataset folders.
- Zip archives such as `dataset2.zip`, `seg_dataset2.zip`, and `ground truth.zip`.
- Trained model weights.
- Large generated outputs from experiments or notebooks.

The dashboard uses curated summaries and lightweight figures so it can run after preparation without the original archives, extracted folders, or model weights.

## Run The Dashboard

Install the lightweight app dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Future Work

Future versions can add live inference after the model weights, preprocessing pipeline, label mapping, and model hosting workflow are cleaned up and documented. Those pieces are intentionally outside the v1 runtime path.
