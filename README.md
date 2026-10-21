# CNN-Satellite Image Land-Cover Classification Using Deep Learning

A full-stack deep learning project for classifying land cover in satellite imagery. The backend exposes a FastAPI application for image prediction and explainability, while the frontend provides a dashboard for upload, preview, confidence visualization, and model insights.

## Project goals

- Accept RGB satellite or aerial images
- Predict land-cover class using a CNN / transfer-learning model
- Support EuroSAT-style land-cover labels when available
- Provide API and frontend access for local demo use
- Keep the project structured for GitHub and interview presentations

## Tech stack

- Python 3.12 with TensorFlow CPU
- FastAPI + Uvicorn
- NumPy, Pandas, Matplotlib, Seaborn, scikit-learn
- PIL / Pillow
- React + Vite
- Pytest + FastAPI TestClient

## Repository structure

- backend/: FastAPI app, schemas, services, tests
- frontend/: React dashboard and API client
- scripts/: dataset download, preparation, training, evaluation, prediction helpers
- models/: saved model artifacts and metadata
- data/: dataset storage and manifests
- artifacts/: training/evaluation outputs and saved figures
- docs/: project documentation and interview prep

## Local setup on Windows (PowerShell)

```powershell
cd "C:\CNN-Satellite Image Land-Cover Classification"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

## Backend startup

For predictions, prepare the dataset and train a model first. The API can start without a
model, but prediction requests will return HTTP 503 until `models/best_model.keras` exists.

```powershell
cd "C:\CNN-Satellite Image Land-Cover Classification"
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

- API docs: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/health

## Frontend startup

```powershell
cd "C:\CNN-Satellite Image Land-Cover Classification\frontend"
npm install
npm run dev
```

Open:

- Frontend dashboard: http://localhost:5173

The dashboard accepts RGB satellite images by clicking the upload area or dragging a
file into it. Images up to 10 MB can be previewed without a model. A trained model
is required for land-cover predictions; the smoke-test model is not used for inference.

## Dataset setup

```powershell
cd "C:\CNN-Satellite Image Land-Cover Classification"
.\.venv\Scripts\Activate.ps1
python -m scripts.download_dataset --output data\EuroSAT.zip --extract-to data
python -m scripts.prepare_dataset --dataset-dir data --output-dir data/processed
```

If the automatic download fails, use the official EuroSAT site and save the archive under the project’s data folder, then rerun the same commands.

## Model training and evaluation

```powershell
cd "C:\CNN-Satellite Image Land-Cover Classification"
.\.venv\Scripts\Activate.ps1
python -m scripts.train_model --model-type baseline --epochs 8 --fine-tune-epochs 0 --batch-size 64 --image-size 64 --seed 42
python -m scripts.evaluate_model --model-path models\best_model.keras
```

For CPU-friendly reproducibility on Windows, set TensorFlow's thread limits before training:

```powershell
$env:TF_NUM_INTRAOP_THREADS = "2"
$env:TF_NUM_INTEROP_THREADS = "1"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
```

The default `baseline` is a compact CNN designed for EuroSAT's native 64-pixel RGB tiles
and CPU training. Training uses only the manifest's training and validation splits.
Validation-based early stopping selects model weights; probability calibration is fit using
the validation split only. The untouched test split is evaluated separately
by the second command. Evaluation writes per-class precision/recall/F1, the confusion
matrix, calibrated confidence statistics, and example predictions to
`models\evaluation_metrics.json`.

Latest measured run (8 epochs, 16,200 training images): validation accuracy 70.35%;
held-out test accuracy 71.72%, macro F1 70.06%, and expected calibration error 2.36%
after validation-only temperature scaling. This is still below the requested 95% target.
These are results for this EuroSAT split and model checkpoint, not a guarantee for other
imagery. Higher calibrated confidence is not the same as higher accuracy.

For a short smoke test only, run:

```powershell
python -m scripts.train_model --model-type baseline --epochs 1 --batch-size 32 --quick
```

Quick mode samples at most 100 training images per class, does not fine-tune, and is not
suitable for reporting final model quality. Training saves the selected model as
`models\best_model.keras`. Class order, preprocessing, and input size are recorded with the
model and reused by evaluation and inference. Evaluation results are shown in the dashboard
only when their recorded model hash matches the currently loaded model. Prediction
percentages use the validation-fitted temperature calibration when available; they are
not guarantees of correctness.

Run project scripts from the repository root with `python -m scripts.<module>`. Module mode keeps the project root on Python's import path so scripts can resolve shared backend and script modules.

## Smoke-test workflow

Run the complete automated backend test suite:

```powershell
cd "C:\CNN-Satellite Image Land-Cover Classification"
.\.venv\Scripts\Activate.ps1
python -m pytest backend/tests -q
```

Build the frontend from its own directory:

```powershell
cd "C:\CNN-Satellite Image Land-Cover Classification\frontend"
npm run build
```

## Important notes

- The project is designed around the EuroSAT RGB land-cover dataset.
- Labels are aligned with the real dataset class names and are not invented.
- If a different dataset is used, update the class map, API responses, and evaluation metadata consistently.
- Grad-CAM is an approximation and not proof of causality.
- Predictions on out-of-distribution images may be unreliable.

## Troubleshooting

- Missing TensorFlow packages: reinstall backend requirements inside the project virtual environment.
- NumPy/SciPy compatibility errors: use the pinned versions in backend/requirements.txt.
- Port 8000 already in use: change the port in the uvicorn command or run with a different port.
- Frontend cannot reach backend: verify the Vite proxy or the API base URL in frontend/src/services/api.js.
- Model loading errors: ensure the trained model is saved in the models directory and the filename matches the loaded model reference.

## GitHub usage

1. Initialize git.
2. Add the project files.
3. Commit the code.
4. Create a repository in GitHub.
5. Push the branch.

```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-url>
git push -u origin main
```

## Resume-ready summary

This project demonstrates deep learning, model serving, frontend integration, and real-world deployment patterns for image classification tasks.
