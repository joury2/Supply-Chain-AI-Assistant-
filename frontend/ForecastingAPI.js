// ForecastingAPI.js
/**
 * JavaScript/React client for AI Forecasting API
 * Usage: import { ForecastingAPI } from './ForecastingAPI';
 */

class ForecastingAPI {
  constructor(baseUrl, apiToken) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiToken = apiToken;
    this.sessionId = null;
  }

  async _request(method, endpoint, body = null, isFormData = false) {
    const headers = {
      'Authorization': `Bearer ${this.apiToken}`
    };

    if (!isFormData) {
      headers['Content-Type'] = 'application/json';
    }

    const options = {
      method,
      headers
    };

    if (body) {
      options.body = isFormData ? body : JSON.stringify(body);
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, options);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'API request failed');
    }

    return response.json();
  }

  /**
   * Upload dataset file
   */
  async uploadDataset(file) {
    const formData = new FormData();
    formData.append('file', file);

    const result = await this._request('POST', '/api/v1/upload', formData, true);
    this.sessionId = result.session_id;
    return result;
  }

  /**
   * Analyze dataset
   */
  async analyzeDataset() {
    if (!this.sessionId) {
      throw new Error('No dataset uploaded');
    }

    return this._request('POST', '/api/v1/analyze', {
      session_id: this.sessionId
    });
  }

  /**
   * Create forecast job
   */
  async createForecast(horizon, businessContext = null) {
    if (!this.sessionId) {
      throw new Error('No dataset uploaded');
    }

    const payload = {
      session_id: this.sessionId,
      horizon
    };

    if (businessContext) {
      payload.business_context = businessContext;
    }

    const result = await this._request('POST', '/api/v1/forecast', payload);
    return result.job_id;
  }

  /**
   * Get forecast status
   */
  async getForecastStatus(jobId) {
    return this._request('GET', `/api/v1/forecast/status/${jobId}`);
  }

  /**
   * Wait for forecast to complete (with polling)
   */
  async waitForForecast(jobId, timeout = 300000, pollInterval = 2000) {
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const status = await this.getForecastStatus(jobId);

      if (status.status === 'completed') {
        return status.result;
      } else if (status.status === 'failed') {
        throw new Error(status.error || 'Forecast failed');
      }

      // Wait before polling again
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }

    throw new Error('Forecast timeout');
  }

  /**
   * Chat with AI assistant
   */
  async chat(message) {
    if (!this.sessionId) {
      throw new Error('No dataset uploaded');
    }

    return this._request('POST', '/api/v1/chat', {
      message,
      session_id: this.sessionId
    });
  }

  /**
   * List available models
   */
  async listModels() {
    return this._request('GET', '/api/v1/models');
  }

  /**
   * Delete current session
   */
  async deleteSession() {
    if (!this.sessionId) {
      return;
    }

    await this._request('DELETE', `/api/v1/session/${this.sessionId}`);
    this.sessionId = null;
  }

  /**
   * Health check
   */
  async healthCheck() {
    return this._request('GET', '/health');
  }
}

// ============================================================================
// React Hook Example
// ============================================================================

/**
 * Custom React hook for forecasting
 */
export function useForecastingAPI(baseUrl, apiToken) {
  const [api] = React.useState(() => new ForecastingAPI(baseUrl, apiToken));
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);

  const uploadDataset = async (file) => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.uploadDataset(file);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const analyzeDataset = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.analyzeDataset();
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const createForecast = async (horizon, businessContext) => {
    setLoading(true);
    setError(null);
    try {
      const jobId = await api.createForecast(horizon, businessContext);
      const result = await api.waitForForecast(jobId);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const chat = async (message) => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.chat(message);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return {
    api,
    loading,
    error,
    uploadDataset,
    analyzeDataset,
    createForecast,
    chat
  };
}

// ============================================================================
// React Component Example
// ============================================================================

/**
 * Example React component
 */
export function ForecastingApp() {
  const { uploadDataset, analyzeDataset, createForecast, loading, error } = 
    useForecastingAPI('http://localhost:8000', 'your-token-here');

  const [uploadResult, setUploadResult] = React.useState(null);
  const [analysisResult, setAnalysisResult] = React.useState(null);
  const [forecastResult, setForecastResult] = React.useState(null);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      try {
        const result = await uploadDataset(file);
        setUploadResult(result);
        console.log('Upload successful:', result);
      } catch (err) {
        console.error('Upload failed:', err);
      }
    }
  };

  const handleAnalyze = async () => {
    try {
      const result = await analyzeDataset();
      setAnalysisResult(result);
      console.log('Analysis complete:', result);
    } catch (err) {
      console.error('Analysis failed:', err);
    }
  };

  const handleForecast = async () => {
    try {
      const result = await createForecast(30);
      setForecastResult(result);
      console.log('Forecast complete:', result);
    } catch (err) {
      console.error('Forecast failed:', err);
    }
  };

  return (
    <div className="forecasting-app">
      <h1>AI Forecasting</h1>

      {/* File Upload */}
      <div className="upload-section">
        <h2>1. Upload Dataset</h2>
        <input 
          type="file" 
          accept=".csv,.xlsx,.xls" 
          onChange={handleFileUpload}
          disabled={loading}
        />
        {uploadResult && (
          <div className="success">
            ✅ Uploaded: {uploadResult.rows} rows, {uploadResult.columns.length} columns
          </div>
        )}
      </div>

      {/* Analysis */}
      {uploadResult && (
        <div className="analysis-section">
          <h2>2. Analyze Dataset</h2>
          <button onClick={handleAnalyze} disabled={loading}>
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
          {analysisResult && (
            <div className="result">
              <p>✅ Status: {analysisResult.status}</p>
              <p>Compatible Models: {analysisResult.compatible_models.length}</p>
              <p>Selected: {analysisResult.selected_model.model_name}</p>
            </div>
          )}
        </div>
      )}

      {/* Forecast */}
      {analysisResult && (
        <div className="forecast-section">
          <h2>3. Generate Forecast</h2>
          <button onClick={handleForecast} disabled={loading}>
            {loading ? 'Forecasting...' : 'Forecast 30 Days'}
          </button>
          {forecastResult && (
            <div className="result">
              <p>✅ Forecast Complete!</p>
              <p>Values: {forecastResult.forecast_data?.values?.length} points</p>
              {/* Add chart here */}
            </div>
          )}
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="error">
          ❌ Error: {error}
        </div>
      )}

      {/* Loading Indicator */}
      {loading && (
        <div className="loading">
          ⏳ Processing...
        </div>
      )}
    </div>
  );
}

export { ForecastingAPI };
export default ForecastingAPI;