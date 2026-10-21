# API documentation

Base URL:

- http://localhost:8000

## Endpoints

### GET /health

Checks backend health and whether a model is loaded.

### GET /classes

Returns the supported class list and labels.

### POST /predict

Input: an uploaded image file.

Output:

- `predicted_class`
- `label`
- `confidence`
- `top_3_predictions`
- `probabilities`
- `inference_time_ms`
- `warning`

### POST /explain

Returns prediction output and a Grad-CAM heatmap when supported.

### GET /model-info

Returns architecture information, model metadata, and evaluation availability.

## Swagger UI

Open http://localhost:8000/docs for interactive API exploration.
