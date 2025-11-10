"""
LightGBM Model Executor (improved debugging + safe preprocessing)
"""

import logging
import joblib
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


@register_executor(ModelType.LIGHTGBM)
class LightGBMExecutor(BaseModelExecutor):
    """LightGBM model executor with proper feature handling and debug logging"""

    def load_model(self, model_path: str) -> bool:
        """Load LightGBM model from file"""
        try:
            self.model = joblib.load(model_path)
            self._is_loaded = True
            logger.info(f"✅ LightGBM model loaded: {model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load LightGBM model: {e}")
            logger.debug(traceback.format_exc())
            return False

    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data has required features"""
        if not isinstance(data, pd.DataFrame):
            logger.error("LightGBM requires DataFrame input")
            return False

        # Log quick diagnostics
        logger.info(f"🔎 Validating input for model '{self.metadata.model_name}': shape={data.shape}")
        logger.debug(f"Columns: {list(data.columns)}")
        logger.debug(f"Sample values:\n{data.head(3).to_dict(orient='list')}")

        # Check required features
        missing_features = []
        for feature in self.metadata.required_features:
            if feature not in data.columns:
                missing_features.append(feature)

        if missing_features:
            logger.warning(f"⚠️ Missing features for {self.metadata.model_name}: {missing_features}")
            # We return True so the pipeline can continue (LightGBM handles missing) but we keep strong logging.
            # If you want to enforce presence, return False here.

        logger.info(f"✅ LightGBM input validated: {data.shape}")
        return True

    def preprocess(self, data: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """LightGBM preprocessing - ensure proper feature order and safe conversions"""
        logger.info("🔧 LightGBM preprocessing start")
        processed_data = data.copy()

        # Keep original columns for debugging
        processed_data.attrs['__original_columns__'] = list(data.columns)

        # Detect candidate date/datetime columns and avoid coercing them to numeric
        datetime_cols = []
        for col in processed_data.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(processed_data[col]) or pd.api.types.is_datetime64tz_dtype(processed_data[col]):
                    datetime_cols.append(col)
                # also treat obvious name matches as datetime candidates
                elif 'date' in col.lower() or 'ds' == col.lower() or 'time' in col.lower():
                    # try conversion if it's string-like
                    try:
                        processed_data[col] = pd.to_datetime(processed_data[col])
                        datetime_cols.append(col)
                    except Exception:
                        # keep as-is (string) but do not coerce to numeric
                        pass
            except Exception:
                # safe fallback
                pass

        # Convert object columns to numeric when appropriate — but skip datetime-like and the target column
        for col in processed_data.select_dtypes(include=['object']).columns:
            if col in datetime_cols or col == self.metadata.target_column:
                continue
            # Try numeric conversion where it makes sense (safe)
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')

        # Fill numeric NaNs with 0 as a safe default (LightGBM tolerates NaN but keep deterministic)
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        processed_data[numeric_cols] = processed_data[numeric_cols].fillna(0)

        # Replace inf values
        processed_data = processed_data.replace([np.inf, -np.inf], 0)

        logger.info(f"   Processed shape: {processed_data.shape}")
        logger.debug(f"   Processed columns: {list(processed_data.columns)}")
        logger.debug(f"   Datetime columns detected: {datetime_cols}")

        return processed_data

    def predict(self, preprocessed_data: pd.DataFrame, horizon: int) -> np.ndarray:
        """Generate predictions with LightGBM"""
        logger.info(f"🤖 LightGBM predicting {horizon} periods for model {self.metadata.model_name}")

        try:
            # If the model expects a particular feature order, try to re-order using metadata.required_features.
            # This is best-effort — if required_features aren't present, we'll proceed with given order.
            try:
                expected = [f for f in self.metadata.required_features if f in preprocessed_data.columns]
                if expected:
                    preprocessed_ordered = preprocessed_data[expected]
                else:
                    preprocessed_ordered = preprocessed_data
            except Exception:
                preprocessed_ordered = preprocessed_data

            # Handle multi-step forecasting when only a single row is provided
            if horizon > 1 and len(preprocessed_ordered) == 1:
                base_prediction = self.model.predict(preprocessed_ordered)
                if isinstance(base_prediction, (list, np.ndarray)):
                    base_val = float(np.array(base_prediction).flatten()[0])
                else:
                    base_val = float(base_prediction)
                predictions = self._generate_multistep_forecast(base_val, horizon)
            else:
                # If model outputs a single scalar per row, we flatten to 1D array
                preds = self.model.predict(preprocessed_ordered)
                predictions = np.array(preds).flatten()

            logger.info(f"✅ LightGBM predictions generated: {len(predictions)}")
            # attach sample input for debugging in metadata (not returned here; used in postprocess)
            preprocessed_data.attrs['__pred_sample__'] = preprocessed_data.head(3).to_dict(orient='list')
            return predictions

        except Exception as e:
            logger.error(f"❌ LightGBM prediction failed: {e}")
            logger.debug(traceback.format_exc())

            # If metadata requests debug dumping, write inputs + stacktrace
            try:
                if getattr(self.metadata, 'debug', False) or (isinstance(getattr(self.metadata, 'version', None), str) and 'debug' in self.metadata.version.lower()):
                    debug_dir = '/tmp/lgbm_executor_debug'
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_payload = {
                        'model_name': self.metadata.model_name,
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                        'input_columns': list(preprocessed_data.columns) if isinstance(preprocessed_data, pd.DataFrame) else None,
                        'input_sample': preprocessed_data.head(5).to_dict(orient='list') if isinstance(preprocessed_data, pd.DataFrame) else None
                    }
                    fname = os.path.join(debug_dir, f"debug_{self.metadata.model_name}_{int(pd.Timestamp.now().timestamp())}.json")
                    with open(fname, 'w') as fh:
                        json.dump(debug_payload, fh, default=str, indent=2)
                    logger.warning(f"🔍 Wrote debug dump: {fname}")
            except Exception:
                logger.debug("Failed to write debug dump")

            # Re-raise to allow upper layers to capture and surface the error
            raise


    def postprocess(self, raw_predictions: np.ndarray) -> ForecastResult:
        """Transform LightGBM predictions to standardized format"""
        logger.info("🔄 Postprocessing LightGBM predictions")

        predictions_array = np.array(raw_predictions).flatten()
        predictions = predictions_array.tolist()  # Convert to list for ForecastResult
        
        # confidence intervals
        confidence_lower, confidence_upper = self._calculate_lightgbm_confidence(predictions_array)
        
        # Convert confidence intervals to list of tuples as expected by ForecastResult
        confidence_intervals = [
            (float(lower), float(upper)) 
            for lower, upper in zip(confidence_lower, confidence_upper)
        ]
        
        # feature importance
        feature_importance = self._get_feature_importance()

        # create a debug id for traceability
        debug_id = f"{self.metadata.model_name}-{int(pd.Timestamp.now().timestamp())}"

        metadata = {
            'model_name': self.metadata.model_name,
            'model_type': 'lightgbm',
            'timestamp': pd.Timestamp.now().isoformat(),
            'feature_importance': feature_importance,
            'executor_version': '1.0',
            'debug_id': debug_id
        }

        # If preprocess attached a sample, include it
        # find a way to attach sample if available (best effort)
        try:
            # Some upstream code may have set attrs on DataFrame and we still have reference in memory - best effort only
            # Not guaranteed — but we try to include if available.
            metadata['input_sample'] = getattr(self, '_last_input_sample', None)
        except Exception:
            pass

        return ForecastResult(
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            model_name=self.metadata.model_name,
            metadata=metadata
        )

    def _generate_multistep_forecast(self, base_prediction: float, horizon: int) -> np.ndarray:
        """Generate multi-step forecast from single prediction (simplified)"""
        # In production, implement proper recursive forecasting / simulation
        return np.array([base_prediction * (1 + i * 0.01) for i in range(horizon)])


    def _calculate_lightgbm_confidence(self, predictions: np.ndarray) -> tuple:
        """Calculate confidence intervals for LightGBM predictions"""
        # LightGBM doesn't provide prediction intervals by default
        uncertainty = np.abs(predictions) * 0.1  # 10% uncertainty as heuristic
        confidence_lower = np.maximum(predictions - uncertainty, 0)
        confidence_upper = predictions + uncertainty
        return confidence_lower, confidence_upper

    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Extract feature importance from LightGBM model"""
        try:
            # scikit-learn wrappers and LightGBM native both may expose these attributes
            if hasattr(self.model, 'feature_importances_'):
                # if feature names exist, try to use them
                names = None
                if hasattr(self.model, 'feature_name_'):
                    names = getattr(self.model, 'feature_name_', None)
                elif hasattr(self.model, 'booster_') and hasattr(self.model.booster_, 'feature_name'):
                    names = self.model.booster_.feature_name()
                if names is not None:
                    importance = getattr(self.model, 'feature_importances_', None)
                    if importance is not None and len(names) == len(importance):
                        importance_dict = dict(zip(names, importance))
                        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            # fallback: for sklearn pipeline with wrapped estimator - cannot reliably extract
        except Exception:
            logger.debug("Failed to extract feature importance", exc_info=True)
        return None
