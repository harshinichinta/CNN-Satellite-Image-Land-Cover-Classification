import { useEffect, useMemo, useState } from 'react';
import { explainImage, getClasses, getHealth, getModelInfo, predictImage } from './services/api';

const defaultClassList = [
  'AnnualCrop',
  'Forest',
  'HerbaceousVegetation',
  'Highway',
  'Industrial',
  'Pasture',
  'PermanentCrop',
  'Residential',
  'River',
  'SeaLake',
];

function App() {
  const [backendStatus, setBackendStatus] = useState(null);
  const [classes, setClasses] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [result, setResult] = useState(null);
  const [explainResult, setExplainResult] = useState(null);
  const [explainError, setExplainError] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    loadInitialData();
  }, []);

  useEffect(() => {
    if (!selectedFile) return;
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  async function loadInitialData() {
    const results = await Promise.allSettled([getHealth(), getClasses(), getModelInfo()]);
    const [health, classData, info] = results;
    const failures = results
      .filter((item) => item.status === 'rejected')
      .map((item) => item.reason?.message || 'A dashboard service could not be reached.');

    if (health.status === 'fulfilled') setBackendStatus(health.value);
    if (classData.status === 'fulfilled') setClasses(classData.value.classes || []);
    if (info.status === 'fulfilled') setModelInfo(info.value);
    if (failures.length) {
      setError(failures.join(' '));
    }
  }

  function selectImage(file) {
    if (!file) return;
    const extension = file.name.split('.').pop()?.toLowerCase();
    const supportedExtensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp'];
    if (!supportedExtensions.includes(extension)) {
      setError('Use a JPG, PNG, BMP, TIFF, or WebP image.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('Image must be 10 MB or smaller.');
      return;
    }

    setSelectedFile(file);
    setResult(null);
    setExplainResult(null);
    setExplainError('');
    setError('');
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    selectImage(event.dataTransfer.files?.[0]);
  }

  async function handlePredict() {
    if (!selectedFile) {
      setError('Please choose an image before predicting.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);
    setExplainResult(null);
    setExplainError('');

    try {
      const prediction = await predictImage(selectedFile);
      setResult(prediction);

      try {
        const explain = await explainImage(selectedFile);
        setExplainResult(explain);
      } catch (err) {
        setExplainResult(null);
        setExplainError(err.message || 'Grad-CAM explanation is unavailable.');
      }
    } catch (err) {
      setError(err.message || 'Prediction failed.');
    } finally {
      setLoading(false);
    }
  }

  const topThree = useMemo(() => result?.top_3_predictions || [], [result]);
  const modelLoaded = backendStatus?.model_loaded ?? false;

  return (
    <div className="app-shell" id="overview">
      <aside className="sidebar">
        <div className="brand-box">
          <div className="brand-lockup">
            <span className="brand-mark" aria-hidden="true">EO</span>
            <span className="eyebrow">Earth observation lab</span>
          </div>
          <h1>TerraScope</h1>
          <p className="brand-caption">Satellite intelligence platform</p>
          <nav className="side-nav" aria-label="Workspace navigation">
            <a href="#overview" className="side-nav-link is-current">Overview</a>
            <a href="#predict" className="side-nav-link">Image analysis</a>
            <a href="#insights" className="side-nav-link">Model insights</a>
          </nav>
        </div>

        <div className="panel">
          <h3>System status</h3>
          <div className="status-row">
            <span className="dot" data-active={backendStatus?.status === 'ok'} />
            <span>{backendStatus?.status === 'ok' ? 'API connected' : 'Checking API'}</span>
          </div>
          <div className="status-row model-status-row">
            <span className="dot" data-active={modelLoaded} />
            <span>{modelLoaded ? 'Inference model ready' : 'Model not loaded'}</span>
          </div>
          {backendStatus && !modelLoaded && (
            <p className="warning-box">
              Add a trained model to <code>models</code> to enable predictions.
            </p>
          )}
        </div>

        <div className="panel">
          <div className="panel-heading">
            <h3>Land-cover classes</h3>
            <span className="count-badge">{classes.length || defaultClassList.length}</span>
          </div>
          <ul className="class-list">
            {(classes.length ? classes : defaultClassList.map((name) => ({ class_name: name, label: name }))).map((item) => (
              <li key={item.class_name || item}>
                <span className="class-swatch" aria-hidden="true" />
                {item.label || item.class_name || item}
              </li>
            ))}
          </ul>
        </div>

        <div className="panel">
          <h3>Model details</h3>
          {modelInfo ? (
            <ul className="meta-list">
              <li>Architecture: {modelInfo.architecture_name}</li>
              <li>Image size: {modelInfo.image_size?.join(' × ') || '224 × 224'}</li>
              <li>Model file: {modelInfo.model_file || 'Not loaded'}</li>
              {modelInfo.training_samples != null && (
                <li>Training images: {modelInfo.training_samples.toLocaleString()}</li>
              )}
              {modelInfo.evaluation_available ? (
                <>
                  <li>Held-out test accuracy: {(modelInfo.evaluation_accuracy * 100).toFixed(1)}%</li>
                  <li>Test macro F1: {(modelInfo.evaluation_macro_f1 * 100).toFixed(1)}%</li>
                  <li>Test images: {modelInfo.evaluation_test_samples.toLocaleString()}</li>
                  {modelInfo.evaluation_low_confidence_fraction != null && (
                    <li>
                      Test images below {(modelInfo.confidence_warning_threshold * 100).toFixed(0)}% score:{' '}
                      {(modelInfo.evaluation_low_confidence_fraction * 100).toFixed(1)}%
                    </li>
                  )}
                </>
              ) : (
                <li>Held-out evaluation: Not available for this model</li>
              )}
              {modelInfo.quick_training && (
                <li className="warning-text">Quick-training model; not intended for reliable evaluation</li>
              )}
            </ul>
          ) : (
            <p>Waiting for model metadata.</p>
          )}
        </div>
        <p className="sidebar-footnote">EUROSAT RGB <span>·</span> LOCAL INFERENCE</p>
      </aside>

      <main className="main-panel">
        <div className="topbar">
          <div className="breadcrumb">WORKSPACE <span>/</span> OVERVIEW</div>
          <a className="docs-link" href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">
            API documentation <span aria-hidden="true">↗</span>
          </a>
        </div>

        <header className="hero">
          <div>
            <p className="eyebrow">Geospatial machine learning</p>
            <h2>Understand the land beneath the pixels.</h2>
            <p className="hero-description">
              Classify satellite imagery into meaningful land-cover categories with an explainable
              computer-vision workflow.
            </p>
          </div>
          <div className="hero-badge">
            <span className="hero-badge-icon" aria-hidden="true">⌖</span>
            <span><strong>EUROSAT RGB</strong><small>10 land-cover categories</small></span>
          </div>
        </header>

        <section className="overview-metrics" aria-label="Project overview">
          <article className="overview-metric panel">
            <span className="metric-icon" aria-hidden="true">◈</span>
            <div><span className="metric-label">CLASSIFICATION SCOPE</span><strong>10 classes</strong></div>
          </article>
          <article className="overview-metric panel">
            <span className="metric-icon" aria-hidden="true">▧</span>
            <div><span className="metric-label">INPUT IMAGERY</span><strong>RGB satellite tiles</strong></div>
          </article>
          <article className="overview-metric panel">
            <span className="metric-icon" aria-hidden="true">◎</span>
            <div><span className="metric-label">INFERENCE ENGINE</span><strong>{modelLoaded ? 'Ready' : 'Awaiting model'}</strong></div>
          </article>
        </section>

        <section className="upload-card panel" id="predict">
          <div className="section-heading">
            <div>
              <p className="eyebrow">01 / Image analysis</p>
              <h3>Start with a satellite image</h3>
            </div>
            <span className="input-spec">RGB · MAX 10 MB</span>
          </div>
          <label
            className={`upload-box${isDragging ? ' is-dragging' : ''}`}
            htmlFor="image-upload"
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget)) setIsDragging(false);
            }}
            onDrop={handleDrop}
          >
            <input
              id="image-upload"
              type="file"
              accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp,image/jpeg,image/png,image/bmp,image/tiff,image/webp"
              onChange={(event) => selectImage(event.target.files?.[0])}
            />
            <span>
              {selectedFile
                ? `${selectedFile.name} — click or drop to replace`
                : 'Click or drag and drop a satellite image here'}
            </span>
            <span className="upload-hint">RGB JPG, PNG, TIFF, BMP, or WebP · up to 10 MB</span>
          </label>

          <div className="actions">
            <button onClick={handlePredict} disabled={loading || !selectedFile || !modelLoaded}>
              {loading ? 'Analyzing image...' : 'Run classification'}
            </button>
            <button className="secondary" onClick={() => {
              setSelectedFile(null);
              setResult(null);
              setExplainResult(null);
              setExplainError('');
              setError('');
              setIsDragging(false);
            }}>
              Clear workspace
            </button>
          </div>

          {error && <div className="error-banner">{error}</div>}
          {selectedFile && !modelLoaded && (
            <div className="warning-banner">
              Image ready to preview. A trained model is needed to run classification.
            </div>
          )}
        </section>

        <section className="content-grid" id="insights">
          <div className="panel preview-panel">
            <div className="panel-heading">
              <h3>Image preview</h3>
              <span className="section-index">02</span>
            </div>
            {previewUrl ? (
              <img src={previewUrl} alt="Selected satellite image preview" className="preview-image" />
            ) : (
              <div className="empty-state">No image selected.</div>
            )}
          </div>

          <div className="panel result-panel">
            <div className="panel-heading">
              <h3>Classification result</h3>
              <span className="section-index">03</span>
            </div>
            {result ? (
              <>
                <div className="prediction-title">{result.label}</div>
                <div className="confidence-box">Top-class probability: {(result.confidence * 100).toFixed(1)}%</div>
                <div className="small-label">
                  {result.probability_method === 'validation_temperature_scaled'
                    ? 'Calibrated on validation data; still not a guarantee of correctness.'
                    : 'Raw model softmax score; not a guarantee of correctness.'}
                </div>
                {result.warning && <div className="warning-banner">{result.warning}</div>}
                {explainError && <div className="warning-banner">Classification succeeded; explanation unavailable: {explainError}</div>}
                <div className="small-label">Inference time: {result.inference_time_ms.toFixed(2)} ms</div>

                <div className="top3-grid">
                  {topThree.map((item) => (
                    <div key={item.class_name} className="metric-card">
                      <div className="metric-rank">#{item.rank}</div>
                      <div className="metric-name">{item.label}</div>
                      <div className="metric-value">{(item.probability * 100).toFixed(1)}%</div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="empty-state">Prediction output will appear here.</div>
            )}
          </div>
        </section>

        <section className="panel probability-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">04 / Model output</p>
              <h3>Class probability distribution</h3>
            </div>
            <span className="small-label">SOFTMAX SCORES</span>
          </div>
          {result ? (
            <div className="bar-list">
              {Object.entries(result.probabilities).map(([name, value]) => (
                <div key={name} className="bar-row">
                  <span>{classes.find((item) => item.class_name === name)?.label || name}</span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${Math.max(2, value * 100)}%` }} />
                  </div>
                  <strong>{(value * 100).toFixed(1)}%</strong>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">Probability chart will appear after prediction.</div>
          )}
        </section>

        {explainResult && explainResult.explainability_supported && explainResult.heatmap_base64 && (
          <section className="panel explainability-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">05 / Explainability</p>
                <h3>Visual model attention</h3>
              </div>
              <span className="input-spec">GRAD-CAM</span>
            </div>
            <div className="comparison-grid">
              <div>
                <p className="small-label">Original</p>
                <img src={previewUrl} alt="Original satellite image" className="mini-image" />
              </div>
              <div>
                <p className="small-label">Heatmap</p>
                <img src={`data:image/png;base64,${explainResult.heatmap_base64}`} alt="Grad-CAM heatmap" className="mini-image" />
              </div>
            </div>
          </section>
        )}
        {explainResult && !explainResult.explainability_supported && (
          <section className="panel explainability-note">
            <p className="eyebrow">05 / Explainability</p>
            <h3>Visual explanation unavailable</h3>
            <p>
              This model completed classification, but does not expose a compatible feature map
              for Grad-CAM. The prediction and class probabilities are still available above.
            </p>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
