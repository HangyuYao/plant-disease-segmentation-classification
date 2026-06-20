# Streamlit Results Dashboard Design

## Goal

Turn the Colab-based plant disease project into a clean GitHub portfolio project centered on a Streamlit research dashboard. Version 1 presents existing results, metrics, figures, masks, and experiment summaries without requiring trained model weights or live inference.

## Current Project Context

The workspace currently contains the original Colab notebook, a Colab-exported Python script, and three data archives:

- `plant_disease_project.ipynb`: main notebook with embedded visual and text outputs.
- `plant_disease_project.py`: exported script with Colab paths and training/evaluation code.
- `dataset2.zip`: 2,400 original images across 12 classes.
- `seg_dataset2.zip`: 2,400 segmented images across 12 classes.
- `ground truth.zip`: 350 lesion mask images across 7 disease classes.

The folder is not yet a git repository. No local model weights or standalone generated result files are present outside the notebook and archives.

## Scope

Version 1 will build a static-results dashboard. It will not train models, run inference, require uploaded images, or require large datasets at runtime.

The dashboard will work from curated lightweight artifacts committed to the repository:

- dataset metadata extracted from zip headers
- selected notebook figures exported as small image assets
- small CSV/JSON metric summaries derived from notebook output
- written explanations of the experimental pipeline and findings

Large archives, extracted datasets, model weights, and bulky generated outputs will be excluded from GitHub.

## User Experience

The Streamlit app will use tabs:

- Overview: project goal, pipeline, key conclusions.
- Dataset: class counts for original, segmented, and mask datasets.
- Experiment 1: original vs leaf-segmented classification comparison.
- Segmentation: mIoU/F1/time comparison and selected mask visual examples.
- Experiment 2: lesion-only masked dataset classification comparison.
- Visual Gallery: curated notebook figures such as masks, overlays, Grad-CAM, and result plots.
- Future Inference: disabled roadmap section explaining what weights/preprocessing/labels would be needed later.

## Architecture

The repository will separate dashboard code from data preparation and project documentation:

- `app/streamlit_app.py` owns the Streamlit UI.
- `src/portfolio_data.py` loads curated CSV/JSON artifacts.
- `src/notebook_assets.py` extracts selected notebook outputs into image assets.
- `src/archive_inventory.py` reads zip metadata without unpacking large datasets.
- `reports/` stores small curated summaries used by the dashboard.
- `assets/figures/` stores selected lightweight portfolio figures.
- `data/README.md` documents dataset sources and why data archives are not committed.

The first implementation will keep inference code out of the runtime path. A future `src/inference/` package can be added later once model weights, class labels, preprocessing, and hosting policy are clean.

## GitHub Exclusions

The `.gitignore` will exclude:

- `*.zip`
- extracted dataset folders
- model weights: `*.pt`, `*.pth`, `*.ckpt`
- large/generated experiment folders
- notebook checkpoints and cache folders
- local Colab/Drive artifacts

## Testing And Verification

The initial implementation will include lightweight tests for:

- archive inventory metadata extraction
- curated report loading
- notebook figure extraction selection behavior

Manual verification will run the Streamlit app locally and confirm it loads without datasets, weights, or Google Colab paths.

## Future Inference Path

Live inference can be added later as an optional tab after these are available:

- committed or externally hosted class label mapping
- documented image preprocessing
- model weights stored outside GitHub, such as GitHub Releases or Hugging Face
- small sample images for smoke testing
- clear fallback when weights are missing

Version 1 will document this path but not implement prediction or mask generation.
