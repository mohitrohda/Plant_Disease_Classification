import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleFile(file);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      handleFile(file);
    }
  };

  const handleFile = (file) => {
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setPrediction(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://13.48.13.229:8000/predict',formData);
      setPrediction(response.data);
    } catch (err) {
      setError('Failed to classify image. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const resetUpload = () => {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="App">
      {/* Animated Background */}
      <div className="background-animation">
        <div className="leaf leaf-1"></div>
        <div className="leaf leaf-2"></div>
        <div className="leaf leaf-3"></div>
      </div>

      {/* Header */}
      <header className="header">
        <div className="logo">
          <svg className="leaf-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L12 7M12 2L9 5M12 2L15 5" stroke="#2d5a27" strokeWidth="2" strokeLinecap="round"/>
            <path d="M12 7C7.5 7 4 10.5 4 15C4 19.5 7.5 23 12 23C16.5 23 20 19.5 20 15C20 10.5 16.5 7 12 7Z" stroke="#2d5a27" strokeWidth="2" fill="#4a9e3f" fillOpacity="0.3"/>
            <path d="M12 12V18M12 12L9 15M12 12L15 15" stroke="#2d5a27" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <h1>Plant Disease Detector</h1>
        </div>
        <p className="subtitle">AI-Powered Plant Health Diagnosis</p>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {!prediction && !loading && (
          <div className="upload-section">
            <div 
              className={`upload-area ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              {!preview ? (
                <>
                  <div className="upload-icon">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M12 16V4M12 4L8 8M12 4L16 8" stroke="#4a9e3f" strokeWidth="2" strokeLinecap="round"/>
                      <path d="M4 16V18C4 19.1046 4.89543 20 6 20H18C19.1046 20 20 19.1046 20 18V16" stroke="#4a9e3f" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                  </div>
                  <h3>Upload Plant Image</h3>
                  <p>Drag & drop or click to select</p>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    id="file-input"
                  />
                  <label htmlFor="file-input" className="upload-button">
                    Choose Image
                  </label>
                  <p className="upload-info">Supports: JPG, PNG, JPEG (Max 10MB)</p>
                </>
              ) : (
                <div className="preview-section">
                  <img src={preview} alt="Plant preview" className="preview-image" />
                  <div className="preview-actions">
                    <button onClick={resetUpload} className="reset-button">
                      Choose Different Image
                    </button>
                    <button onClick={handleSubmit} className="analyze-button">
                      Analyze Disease
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {loading && (
          <div className="loading-section">
            <div className="spinner"></div>
            <h3>Analyzing Plant Health...</h3>
            <p>Our AI is examining the image for disease patterns</p>
            <div className="loading-steps">
              <div className="step">📸 Image Processing</div>
              <div className="step">🔬 Feature Extraction</div>
              <div className="step">🧠 Neural Network Analysis</div>
            </div>
          </div>
        )}

        {prediction && (
          <div className="results-section">
            <div className="results-card">
              <div className="results-header">
                <h2>Diagnosis Complete</h2>
                <button onClick={resetUpload} className="new-analysis-button">
                  New Analysis
                </button>
              </div>
              
              <div className="results-grid">
                <div className="image-preview">
                  <img src={preview} alt="Analyzed plant" />
                </div>
                
                <div className="diagnosis-info">
                  <div className="disease-card">
                    <div className="disease-icon">
                      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 8V12L15 15" stroke="#2d5a27" strokeWidth="2" strokeLinecap="round"/>
                        <circle cx="12" cy="12" r="9" stroke="#2d5a27" strokeWidth="2"/>
                      </svg>
                    </div>
                    <div className="disease-details">
                      <h3>Detected Condition</h3>
                      <p className="disease-name">{prediction.class.replace(/_/g, ' ')}</p>
                      <div className="confidence-bar">
                        <div className="confidence-label">
                          <span>Confidence Score</span>
                          <span className="confidence-value">{(prediction.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="bar-background">
                          <div 
                            className="bar-fill" 
                            style={{ width: `${prediction.confidence * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="recommendations">
                    <h4>Recommendations</h4>
                    <ul>
                      <li>✓ Consult with a local agricultural expert</li>
                      <li>✓ Isolate affected plants to prevent spread</li>
                      <li>✓ Review watering and fertilization practices</li>
                      <li>✓ Consider organic treatment options</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="error-section">
            <div className="error-card">
              <div className="error-icon">⚠️</div>
              <h3>Analysis Failed</h3>
              <p>{error}</p>
              <button onClick={resetUpload} className="try-again-button">
                Try Again
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>Powered by Deep Learning | Plant Disease Detection Model</p>
        <p className="footer-note">For accurate diagnosis, ensure good lighting and clear plant images</p>
      </footer>
    </div>
  );
}

export default App;