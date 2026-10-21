# Methodology

## Dataset

The default project path supports the EuroSAT RGB land-cover dataset. This dataset is suitable for multi-class satellite image classification and provides a realistic benchmark for land-cover tasks.

## Pipeline

1. Download and validate the image archive.
2. Inspect dataset structure and class folders.
3. Analyze class distribution and data quality.
4. Create reproducible train/validation/test splits.
5. Build a baseline CNN and a transfer-learning model.
6. Train, save metadata, and keep preprocessing consistent.
7. Evaluate on the held-out test set.
8. Expose the model through FastAPI.
9. Build a React dashboard for local demonstration.

## Model decisions

- Baseline CNN is useful for quick experiments and low-resource workflows.
- Transfer learning is preferred when stronger feature extraction is needed and the problem is image-based.
- Both models should be evaluated on the same test set without tuning on the test data.

## Explainability

Grad-CAM is a gradient-based approximation of which image regions most influenced the prediction. It is useful for debugging and demonstration, but it is not a causal proof of model behavior.
