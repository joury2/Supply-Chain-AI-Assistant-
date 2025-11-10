# app/services/inference/executors/lstm_executor.py
"""
LSTM Model Executor (improved debugging + robust sequence handling)
"""

import logging
import pickle
import pandas as pd
import numpy as np
import json
import os
import traceback
from typing import Any, Optional, Dict

from app.services.inference.base_model_executor import (
    BaseModelExecutor, ModelMetadata, ForecastResult,
    ModelType, register_executor
)

logger = logging.getLogger(__name__)

@register_executor(ModelType.LSTM)
class LSTMExecutor(BaseModelExecutor):
    """LSTM model executor with robust sequence handling and comprehensive debugging"""
    
    def __init__(self, metadata: ModelMetadata):
        super().__init__(metadata)
        self.preprocessor = None
        self.expected_features = None
        self.expected_feature_count = None  # Expected number of features from model input shape
        self._preprocessing_metadata = {}  # Store preprocessing metadata here
    
    def load_model(self, model_path: str) -> bool:
        """Load LSTM model and preprocessor with enhanced error handling"""
        try:
            import tensorflow as tf
            
            # Load the model
            self.model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"✅ LSTM model loaded: {model_path}")
            
            # Log model architecture
            try:
                self.model.summary(print_fn=logger.debug)
            except:
                logger.debug("   Could not print model summary")
            
            # ✅ Extract expected number of features from model input shape
            if self.model.inputs:
                input_shape = self.model.inputs[0].shape
                if len(input_shape) >= 2:
                    # LSTM input shape is (batch, timesteps, features)
                    # The last dimension is the number of features
                    self.expected_feature_count = int(input_shape[-1])
                    logger.info(f"✅ Model expects {self.expected_feature_count} input features")
                else:
                    self.expected_feature_count = None
                    logger.warning("⚠️ Could not determine expected feature count from model")
            else:
                self.expected_feature_count = None
                logger.warning("⚠️ Model has no input layer")
            
            # Try to load preprocessor
            preprocessor_path = model_path.replace('.h5', '_preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
                logger.info("✅ LSTM preprocessor loaded")
                
                # Extract expected features from preprocessor if available
                if hasattr(self.preprocessor, 'feature_names_in_'):
                    self.expected_features = list(self.preprocessor.feature_names_in_)
                    logger.info(f"✅ Preprocessor expects {len(self.expected_features)} features: {self.expected_features}")
            else:
                logger.warning("⚠️ LSTM preprocessor not found")
            
            self._is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load LSTM model: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate LSTM input requirements with feature matching"""
        logger.info(f"🔎 Validating LSTM input for '{self.metadata.model_name}': shape={data.shape}")
        
        if not isinstance(data, pd.DataFrame):
            logger.error("LSTM requires DataFrame input")
            return False
        
        # Check for required features (may be encoded during preprocessing)
        missing_features = []
        categorical_to_encoded = {
            'Customer': 'Customer_encoded',
            'Location': 'Location_encoded',
            'BusinessType': 'BusinessType_encoded'
        }
        
        # Categorical features that may be One-Hot Encoded (OHE)
        # OHE creates columns like 'Customer_Walmart', 'Location_New York', etc.
        categorical_features = ['Customer', 'Location', 'BusinessType']
        
        # Date-derived features that replace WorkDate during preprocessing
        date_derived_features = ['year', 'month', 'day', 'dayofweek', 'quarter']
        
        for feature in self.metadata.required_features:
            # Check if feature exists as-is
            if feature in data.columns:
                continue
            # Check if feature exists as label-encoded version
            elif feature in categorical_to_encoded and categorical_to_encoded[feature] in data.columns:
                logger.debug(f"   Found label-encoded version of {feature}: {categorical_to_encoded[feature]}")
                continue
            # ✅ FIX: Check if feature exists as One-Hot Encoded (OHE) columns
            # OHE creates columns with pattern: {feature}_{category_value}
            elif feature in categorical_features:
                # Check if any columns start with the feature name followed by underscore
                ohe_columns = [col for col in data.columns if col.startswith(f'{feature}_')]
                if ohe_columns:
                    logger.debug(f"   Found One-Hot Encoded version of {feature}: {len(ohe_columns)} OHE columns ({ohe_columns[:3]}...)")
                    continue
                else:
                    missing_features.append(feature)
            # Check if WorkDate was converted to date-derived features
            elif feature == 'WorkDate':
                # Check if date-derived features are present
                if any(df_feat in data.columns for df_feat in date_derived_features):
                    logger.debug(f"   WorkDate converted to date-derived features: {[f for f in date_derived_features if f in data.columns]}")
                    continue
                else:
                    missing_features.append(feature)
            else:
                missing_features.append(feature)
        
        if missing_features:
            logger.error(f"❌ Missing required LSTM features: {missing_features}")
            logger.error(f"   Available features: {list(data.columns)}")
            return False
        
        # Check feature compatibility with preprocessor
        if self.preprocessor and self.expected_features:
            extra_features = set(data.columns) - set(self.expected_features)
            if extra_features:
                logger.warning(f"⚠️ Extra features in input: {list(extra_features)}")
            
            missing_expected = set(self.expected_features) - set(data.columns)
            if missing_expected:
                logger.warning(f"⚠️ Missing expected features: {list(missing_expected)}")
        
        # Validate data types - all features should be numeric after preprocessing
        # (categorical features are encoded during data preparation)
        numeric_issues = []
        encoded_categorical_features = []
        ohe_categorical_features = []
        
        # Check all columns in data (which may include encoded versions)
        for col in data.columns:
            # Skip if this is a label-encoded categorical feature (already numeric)
            if col.endswith('_encoded'):
                encoded_categorical_features.append(col)
                continue
            
            # ✅ FIX: Check if this is an OHE column (starts with categorical feature name + underscore)
            is_ohe_column = False
            for cat_feature in categorical_features:
                if col.startswith(f'{cat_feature}_'):
                    is_ohe_column = True
                    if cat_feature not in ohe_categorical_features:
                        ohe_categorical_features.append(cat_feature)
                    break
            
            if is_ohe_column:
                # OHE columns should be numeric (binary 0/1)
                if not pd.api.types.is_numeric_dtype(data[col]):
                    numeric_issues.append(col)
                continue
            
            # Check if this column corresponds to a required feature
            # (either directly or as an encoded version, or as a date-derived feature)
            is_required = col in self.metadata.required_features
            if not is_required:
                # Check if this is the label-encoded version of a required feature
                for orig_col, encoded_col in categorical_to_encoded.items():
                    if col == encoded_col and orig_col in self.metadata.required_features:
                        is_required = True
                        break
                # Check if this is a date-derived feature replacing WorkDate
                if not is_required and 'WorkDate' in self.metadata.required_features:
                    if col in date_derived_features:
                        is_required = True
            
            if is_required and not pd.api.types.is_numeric_dtype(data[col]):
                # Attempt conversion to numeric
                try:
                    converted = pd.to_numeric(data[col], errors='coerce')
                    if converted.isna().any():
                        numeric_issues.append(col)
                    else:
                        logger.debug(f"   Successfully converted {col} to numeric")
                except Exception:
                    numeric_issues.append(col)
        
        if numeric_issues:
            logger.error(f"❌ Non-numeric values in numeric features: {numeric_issues}")
            return False
        
        if encoded_categorical_features:
            logger.info(f"   Found {len(encoded_categorical_features)} label-encoded categorical features: {encoded_categorical_features}")
        
        if ohe_categorical_features:
            ohe_column_count = sum(1 for col in data.columns if any(col.startswith(f'{cat}_') for cat in ohe_categorical_features))
            logger.info(f"   Found {len(ohe_categorical_features)} One-Hot Encoded categorical features: {ohe_categorical_features} ({ohe_column_count} total OHE columns)")
        
        logger.info(f"✅ LSTM input validated: {data.shape}")
        # Log data types for available columns (including OHE columns)
        available_cols = [
            col for col in data.columns 
            if col in self.metadata.required_features 
            or col.endswith('_encoded')
            or any(col.startswith(f'{cat}_') for cat in categorical_features)
        ]
        if available_cols:
            logger.debug(f"   Data types for key features ({len(available_cols)} columns): {available_cols[:10]}..." if len(available_cols) > 10 else f"   Data types for key features: {available_cols}")
        return True
    
    def preprocess(self, data: pd.DataFrame, horizon: int) -> np.ndarray:
        """LSTM preprocessing with robust feature engineering and reshaping"""
        logger.info("🔧 LSTM preprocessing start")
        
        processed_data = data.copy()
        
        # Store original for debugging
        original_columns = list(processed_data.columns)
        
        try:
            if self.preprocessor:
                # Use saved preprocessor
                logger.debug("   Using saved preprocessor")
                processed_values = self.preprocessor.transform(processed_data)
                logger.debug(f"   Preprocessor output shape: {processed_values.shape}")
            else:
                # Manual preprocessing
                logger.debug("   Using manual preprocessing")
                processed_values = self._manual_preprocess(processed_data)
            
            # ✅ CRITICAL: Validate and align feature count with model expectations
            if hasattr(self, 'expected_feature_count') and self.expected_feature_count is not None:
                current_feature_count = processed_values.shape[-1] if len(processed_values.shape) >= 2 else processed_values.shape[0]
                if current_feature_count != self.expected_feature_count:
                    logger.warning(f"⚠️ Feature count mismatch: got {current_feature_count}, model expects {self.expected_feature_count}")
                    # If we have a DataFrame, we can try to align features
                    if isinstance(processed_data, pd.DataFrame) and not self.preprocessor:
                        processed_values = self._align_features(processed_data, self.expected_feature_count)
                        logger.info(f"   Aligned features to {processed_values.shape[-1]} (expected: {self.expected_feature_count})")
                    else:
                        logger.error(f"❌ Cannot align features: preprocessor used or data is not DataFrame")
                        raise ValueError(f"Feature count mismatch: {current_feature_count} != {self.expected_feature_count}")
            
            # Reshape for LSTM: (samples, timesteps, features)
            # Assuming we want 1 timestep per sample (simplified approach)
            shape_before_reshape = processed_values.shape
            if len(processed_values.shape) == 2:
                processed_values = processed_values.reshape(
                    (processed_values.shape[0], 1, processed_values.shape[1])
                )
            
            # Final validation after reshape
            if hasattr(self, 'expected_feature_count') and self.expected_feature_count is not None:
                final_feature_count = processed_values.shape[-1]
                if final_feature_count != self.expected_feature_count:
                    logger.error(f"❌ Final feature count mismatch: {final_feature_count} != {self.expected_feature_count}")
                    raise ValueError(f"Feature count mismatch after reshape: {final_feature_count} != {self.expected_feature_count}")
            
            logger.info(f"   LSTM input shape: {processed_values.shape}")
            
            # ✅ FIX: Store preprocessing info in executor instance (numpy arrays don't have .attrs)
            self._preprocessing_metadata = {
                'original_columns': original_columns,
                'preprocessor_used': bool(self.preprocessor),
                'input_shape_before_reshape': shape_before_reshape,
                'input_shape_after_reshape': processed_values.shape,
                'expected_feature_count': getattr(self, 'expected_feature_count', None)
            }
            
            return processed_values
            
        except Exception as e:
            logger.error(f"❌ LSTM preprocessing failed: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def predict(self, preprocessed_data: np.ndarray, horizon: int) -> np.ndarray:
        """Generate predictions with LSTM with comprehensive error handling"""
        logger.info(f"🤖 LSTM predicting {horizon} periods for {self.metadata.model_name}")
        logger.debug(f"   Input shape: {preprocessed_data.shape}")
        
        try:
            # ✅ FIX: For time series forecasting, use only the LAST sample from the input sequence
            # The input has shape (samples, timesteps, features) - we only need the last sample
            # to predict the future, not predictions for all historical data points
            if len(preprocessed_data.shape) == 3:
                # LSTM input: (samples, timesteps, features)
                # Use only the last sample for forecasting
                last_sample = preprocessed_data[-1:]  # Shape: (1, timesteps, features)
                logger.debug(f"   Using last sample for forecasting: {last_sample.shape}")
                predictions = self.model.predict(last_sample, verbose=0)
            else:
                # Fallback: if shape is different, use the last sample anyway
                if len(preprocessed_data.shape) == 2:
                    last_sample = preprocessed_data[-1:]  # Shape: (1, features)
                    logger.debug(f"   Using last sample (2D input): {last_sample.shape}")
                else:
                    last_sample = preprocessed_data
                    logger.debug(f"   Using full input (unexpected shape): {last_sample.shape}")
                predictions = self.model.predict(last_sample, verbose=0)
            
            logger.debug(f"   Raw LSTM predictions shape: {predictions.shape}")
            
            # Flatten predictions to 1D
            predictions = predictions.flatten()
            logger.debug(f"   Flattened predictions shape: {predictions.shape}, length: {len(predictions)}")
            
            # ✅ FIX: Handle horizon - we need exactly 'horizon' predictions
            if len(predictions) == 1:
                # Single prediction - generate multi-step forecast
                base_pred = float(predictions[0])
                predictions = self._generate_multistep_forecast(base_pred, horizon)
                logger.debug(f"   Generated multi-step forecast from single prediction: {len(predictions)} steps")
            elif len(predictions) > horizon:
                # Model returned more predictions than needed - take only the first 'horizon' predictions
                predictions = predictions[:horizon]
                logger.debug(f"   Truncated predictions to horizon: {len(predictions)} steps")
            elif len(predictions) < horizon:
                # Model returned fewer predictions than needed - extend using the last prediction
                logger.warning(f"   Model returned {len(predictions)} predictions, but horizon is {horizon}")
                last_pred = predictions[-1]
                extended = self._generate_multistep_forecast(float(last_pred), horizon - len(predictions))
                predictions = np.concatenate([predictions, extended])
                logger.debug(f"   Extended predictions to horizon: {len(predictions)} steps")
            
            # Ensure we have exactly 'horizon' predictions
            if len(predictions) != horizon:
                logger.warning(f"   Prediction length ({len(predictions)}) != horizon ({horizon}), adjusting...")
                if len(predictions) > horizon:
                    predictions = predictions[:horizon]
                else:
                    # Pad with last prediction
                    last_pred = predictions[-1] if len(predictions) > 0 else 0.0
                    padding = np.full(horizon - len(predictions), last_pred)
                    predictions = np.concatenate([predictions, padding])
            
            logger.info(f"✅ LSTM predictions generated: {len(predictions)} (horizon: {horizon})")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ LSTM prediction failed: {e}")
            
            # Debug dump
            try:
                if getattr(self.metadata, 'debug', False):
                    debug_dir = '/tmp/lstm_executor_debug'
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_payload = {
                        'model_name': self.metadata.model_name,
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                        'input_shape': preprocessed_data.shape,
                        'input_attrs': getattr(preprocessed_data, 'attrs', {}),
                        'model_input_shape': getattr(self.model, 'input_shape', 'unknown')
                    }
                    fname = os.path.join(debug_dir, f"debug_{self.metadata.model_name}_{int(pd.Timestamp.now().timestamp())}.json")
                    with open(fname, 'w') as fh:
                        json.dump(debug_payload, fh, default=str, indent=2)
                    logger.warning(f"🔍 Wrote debug dump: {fname}")
            except Exception as debug_e:
                logger.debug(f"Failed to write debug dump: {debug_e}")
            
            raise
    
    
    def postprocess(self, raw_predictions: np.ndarray) -> ForecastResult:
        """Transform LSTM predictions to standardized format with enhanced metadata"""
        logger.info("🔄 Postprocessing LSTM predictions")
        
        predictions_array = np.array(raw_predictions).flatten()
        predictions = predictions_array.tolist()  # Convert to list for ForecastResult
        
        # Calculate confidence intervals
        confidence_lower, confidence_upper = self._calculate_lstm_confidence(predictions_array)
        
        # Convert confidence intervals to list of tuples as expected by ForecastResult
        confidence_intervals = [
            (float(lower), float(upper)) 
            for lower, upper in zip(confidence_lower, confidence_upper)
        ]
        
        # Create enhanced metadata
        debug_id = f"lstm-{self.metadata.model_name}-{int(pd.Timestamp.now().timestamp())}"
        
        metadata = {
            'model_name': self.metadata.model_name,
            'model_type': 'lstm',
            'timestamp': pd.Timestamp.now().isoformat(),
            'input_shape': getattr(self.model, 'input_shape', 'unknown'),
            'executor_version': '1.0',
            'debug_id': debug_id,
            'preprocessor_used': bool(self.preprocessor),
            'preprocessing_metadata': self._preprocessing_metadata,  # Include stored preprocessing metadata
            'prediction_statistics': {
                'mean': float(np.mean(predictions_array)),
                'std': float(np.std(predictions_array)),
                'min': float(np.min(predictions_array)),
                'max': float(np.max(predictions_array))
            }
        }
        
        return ForecastResult(
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            model_name=self.metadata.model_name,
            metadata=metadata
        )
    
    def _align_features(self, data: pd.DataFrame, expected_count: int) -> np.ndarray:
        """
        Align features to match model's expected feature count.
        This handles cases where OHE creates different numbers of columns than the model expects.
        
        Note: Without knowing the exact feature order from training, we pad missing features
        with zeros at the end. This works when missing features represent categories not present
        in the test data (which should be zero in OHE encoding).
        """
        logger.info(f"   Aligning features: current={data.shape[1]}, expected={expected_count}")
        
        # Get all numeric columns (should be all columns after preprocessing)
        # Preserve original column order (don't sort) to maintain compatibility with model expectations
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        current_count = len(numeric_cols)
        
        logger.debug(f"   Current features ({current_count}): {numeric_cols[:10]}..." if len(numeric_cols) > 10 else f"   Current features: {numeric_cols}")
        
        if current_count == expected_count:
            # Already aligned
            logger.debug("   Features already aligned - no padding needed")
            return data[numeric_cols].values
        
        elif current_count < expected_count:
            # Need to pad with zeros (missing OHE columns from test data)
            padding_count = expected_count - current_count
            logger.warning(f"   Padding {padding_count} missing features with zeros (test data has fewer categories than training data)")
            logger.debug(f"   This likely means test data has fewer unique values in categorical columns")
            
            # Create a DataFrame with the right number of columns
            aligned_data = data[numeric_cols].copy()
            # Add zero columns for missing features
            # Note: Padding at the end assumes missing categories should be represented as zeros,
            # which is correct for OHE encoding when a category is not present
            for i in range(padding_count):
                aligned_data[f'_padding_{i}'] = 0
            
            logger.debug(f"   Aligned feature count: {aligned_data.shape[1]} (expected: {expected_count})")
            return aligned_data.values
        
        else:
            # Have more features than expected - take first N
            truncation_count = current_count - expected_count
            logger.warning(f"   Truncating {truncation_count} extra features (test data has more categories than training data)")
            logger.debug(f"   Taking first {expected_count} features from {current_count} available")
            return data[numeric_cols[:expected_count]].values
    
    def _manual_preprocess(self, data: pd.DataFrame) -> np.ndarray:
        """Manual preprocessing when no preprocessor is available"""
        # ✅ FIX: After OHE, all columns should be numeric (including OHE binary columns)
        # Select all numeric columns (this includes date features, numeric features, and OHE columns)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            logger.error("❌ No numeric columns found in data")
            raise ValueError("No numeric columns available for LSTM preprocessing")
        
        # Use all numeric columns (OHE columns are already binary/numeric)
        processed = data[numeric_cols].copy()
        
        # Convert to numeric (should already be numeric, but ensure)
        for col in processed.columns:
            processed[col] = pd.to_numeric(processed[col], errors='coerce')
        
        # Fill NaN values
        processed = processed.fillna(0)
        
        # Replace inf values
        processed = processed.replace([np.inf, -np.inf], 0)
        
        logger.debug(f"   Manual preprocessing output shape: {processed.shape}")
        logger.debug(f"   Using {len(numeric_cols)} numeric columns (including OHE)")
        logger.debug(f"   Column names: {numeric_cols[:10]}..." if len(numeric_cols) > 10 else f"   Column names: {numeric_cols}")
        
        return processed.values
    
    def _generate_multistep_forecast(self, base_prediction: float, horizon: int) -> np.ndarray:
        """Generate multi-step forecast from single prediction"""
        # Add slight decay for more realistic multi-step forecast
        return np.array([base_prediction * (0.98 ** i) for i in range(horizon)])
    
    def _calculate_lstm_confidence(self, predictions: np.ndarray) -> tuple:
        """Calculate confidence intervals for LSTM predictions"""
        # Higher uncertainty for neural networks
        uncertainty = np.abs(predictions) * 0.15
        
        confidence_lower = np.maximum(predictions - uncertainty, 0)
        confidence_upper = predictions + uncertainty
        
        return confidence_lower, confidence_upper