# Data Notes

Full datasets are not committed to GitHub. The project is designed to keep large archives, extracted data folders, trained weights, and large generated outputs out of version control.

Expected local archives for reproducibility work:

- `dataset2.zip`
- `seg_dataset2.zip`
- `ground truth.zip`

The Streamlit dashboard does not need these archives at runtime once the small curated summaries and figures have been prepared. Version 1 uses the lightweight files in `reports/` and `assets/` so the dashboard can be viewed without full datasets or model weights.
