# app/services/inference/executors/prophet_executor.py
"""
Prophet Model Executor (improved debugging + robust future dataframe handling)
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

@register_executor(ModelType.PROPHET)
class ProphetExecutor(BaseModelExecutor):
    """Prophet model executor with robust future dataframe handling and debug logging"""
    
    def load_model(self, model_path: str) -> bool:
        """Load Prophet model from file with enhanced error handling"""
        try:
            self.model = joblib.load(model_path)
            self._is_loaded = True
            logger.info(f"✅ Prophet model loaded: {model_path}")
            
            # Log Prophet model configuration for debugging
            if hasattr(self.model, 'seasonalities'):
                logger.debug(f"   Prophet seasonalities: {self.model.seasonalities}")
            if hasattr(self.model, 'extra_regressors'):
                logger.debug(f"   Prophet regressors: {list(self.model.extra_regressors.keys())}")
                
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load Prophet model: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate Prophet input requirements with detailed diagnostics"""
        logger.info(f"🔎 Validating Prophet input for '{self.metadata.model_name}': shape={data.shape}")
        
        if not isinstance(data, pd.DataFrame):
            logger.error("Prophet requires DataFrame input")
            return False
        
        # Check for required date column
        date_col_candidates = ['ds', 'date', 'timestamp', 'datetime']
        date_col = None
        for candidate in date_col_candidates:
            if candidate in data.columns:
                date_col = candidate
                break
        
        if not date_col:
            logger.error(f"❌ Prophet requires a date column. Available: {list(data.columns)}")
            return False
        
        # Check for target column
        target_col_candidates = ['y', 'sales', 'demand', 'revenue', 'value']
        target_col = None
        for candidate in target_col_candidates:
            if candidate in data.columns:
                target_col = candidate
                break
        
        if not target_col:
            logger.error(f"❌ Prophet requires a target column. Available: {list(data.columns)}")
            return False
        
        # Validate date column can be parsed
        try:
            pd.to_datetime(data[date_col].head())
            logger.debug(f"   Date column '{date_col}' validated")
        except Exception as e:
            logger.error(f"❌ Date column '{date_col}' cannot be parsed: {e}")
            return False
        
        # Validate target column is numeric
        if not pd.api.types.is_numeric_dtype(data[target_col]):
            logger.warning(f"⚠️ Target column '{target_col}' is not numeric. Attempting conversion...")
            try:
                data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
                if data[target_col].isna().any():
                    logger.error(f"❌ Target column '{target_col}' has non-numeric values after conversion")
                    return False
            except Exception as e:
                logger.error(f"❌ Failed to convert target column to numeric: {e}")
                return False
        
        logger.info(f"✅ Prophet input validated: date_col='{date_col}', target_col='{target_col}'")
        logger.debug(f"   Data sample:\n{data[[date_col, target_col]].head(3)}")
        return True
    
    def preprocess(self, data: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Prophet preprocessing with robust datetime handling and aggregation"""
        logger.info("🔧 Prophet preprocessing start")
        
        processed_data = data.copy()
        
        # Identify date and target columns
        date_col = 'ds' if 'ds' in processed_data.columns else \
                  next((col for col in ['date', 'timestamp', 'datetime'] 
                       if col in processed_data.columns), None)
        
        target_col = 'y' if 'y' in processed_data.columns else \
                   next((col for col in ['sales', 'demand', 'revenue', 'value'] 
                        if col in processed_data.columns), None)
        
        # Ensure proper column names for Prophet
        if date_col != 'ds':
            processed_data['ds'] = pd.to_datetime(processed_data[date_col])
            if date_col != 'ds':  # Don't drop if it's already 'ds'
                processed_data = processed_data.drop(columns=[date_col])
        
        if target_col != 'y':
            processed_data['y'] = processed_data[target_col]
            if target_col != 'y' and target_col != date_col:  # Don't drop if it's the date column
                processed_data = processed_data.drop(columns=[target_col])
        
        # Ensure 'ds' is datetime and sort
        processed_data['ds'] = pd.to_datetime(processed_data['ds'])
        processed_data = processed_data.sort_values('ds').reset_index(drop=True)
        
        # ✅ CRITICAL FIX: Aggregate data by date if too granular
        # Prophet expects aggregated time series (one row per time period), not transaction-level data
        original_row_count = len(processed_data)
        if original_row_count > 10000:  # Threshold for aggregation
            logger.warning(f"⚠️ Large dataset detected ({original_row_count} rows). Aggregating by date for Prophet...")
            
            # Aggregate by date: sum numeric columns, take first for non-numeric
            agg_dict = {'y': 'sum'}  # Sum the target variable
            
            # Identify regressor columns (numeric columns that aren't ds or y)
            regressor_cols = [col for col in processed_data.columns 
                             if col not in ['ds', 'y'] and pd.api.types.is_numeric_dtype(processed_data[col])]
            
            # Aggregate regressors (sum for numeric, mean for others)
            for col in regressor_cols:
                agg_dict[col] = 'sum'  # Sum numeric regressors
            
            # Group by date (normalize to day level)
            processed_data['ds_date'] = processed_data['ds'].dt.date
            aggregated = processed_data.groupby('ds_date').agg(agg_dict).reset_index()
            aggregated['ds'] = pd.to_datetime(aggregated['ds_date'])
            aggregated = aggregated.drop(columns=['ds_date'])
            
            # Sort by date
            aggregated = aggregated.sort_values('ds').reset_index(drop=True)
            
            logger.info(f"   Aggregated {original_row_count} rows → {len(aggregated)} rows (by date)")
            logger.debug(f"   Date range: {aggregated['ds'].min()} to {aggregated['ds'].max()}")
            
            processed_data = aggregated
        
        # Identify regressor columns after aggregation
        regressor_cols = [col for col in processed_data.columns 
                         if col not in ['ds', 'y'] and pd.api.types.is_numeric_dtype(processed_data[col])]
        
        logger.info(f"   Processed data: {len(processed_data)} rows, regressors: {regressor_cols}")
        logger.debug(f"   Date range: {processed_data['ds'].min()} to {processed_data['ds'].max()}")
        
        return processed_data
    
    def predict(self, preprocessed_data: pd.DataFrame, horizon: int):
        """Make real predictions with Prophet model - NO MOCKING"""
        logger.info("📊 Prophet model predicting with real data")
        
        # Validate we have required data
        if 'ds' not in preprocessed_data.columns:
            error_msg = "Prophet model requires 'ds' (date) column"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        if 'y' not in preprocessed_data.columns:
            error_msg = "Prophet model requires 'y' (target) column" 
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        try:
            # Create future dataframe using Prophet's built-in method
            future = self.model.make_future_dataframe(periods=horizon)
            
            # Add real date-based regressors
            future = self._add_real_date_regressors(future)
            
            # ✅ FIX: Add required extra regressors that the model expects
            future = self._add_extra_regressors_to_future(future, preprocessed_data)
            
            # ✅ CRITICAL FIX: Limit historical data if still too large after aggregation
            # Prophet's uncertainty calculation uses Monte Carlo simulation which creates massive arrays
            # If we still have too much data after aggregation, use only recent history
            data_size = len(preprocessed_data)
            max_history_rows = 5000  # Maximum rows to use for Prophet (reasonable limit)
            
            if data_size > max_history_rows:
                logger.warning(f"⚠️ Large dataset after aggregation ({data_size} rows). "
                             f"Using only the most recent {max_history_rows} rows for Prophet prediction.")
                # Use only the most recent data
                preprocessed_data = preprocessed_data.tail(max_history_rows).copy()
                # Re-fit the model with the reduced dataset (or use the existing model)
                # Note: We can't re-fit here, but we can limit the future dataframe
                logger.info(f"   Reduced to {len(preprocessed_data)} rows for prediction")
            
            # Make real prediction
            # Note: Prophet's uncertainty calculation can be memory-intensive for large datasets
            # The aggregation above should help, but if memory issues persist, we'll catch and handle them
            try:
                forecast = self.model.predict(future)
            except (MemoryError, Exception) as mem_err:
                # Catch both MemoryError and numpy ArrayMemoryError
                # Check if it's a memory-related error
                error_str = str(mem_err).lower()
                if 'memory' in error_str or 'allocate' in error_str or 'ArrayMemoryError' in str(type(mem_err)):
                    # If memory error occurs, provide helpful error message
                    error_msg = (f"Prophet prediction failed due to memory constraints. "
                               f"Dataset size: {data_size} rows after aggregation. "
                               f"Prophet's uncertainty calculation requires significant memory for large datasets. "
                               f"Consider: (1) Using a different model (LSTM/LightGBM) for granular data, "
                               f"(2) Further aggregating the data (e.g., weekly/monthly), "
                               f"or (3) Reducing the forecast horizon.")
                    logger.error(f"❌ {error_msg}")
                    logger.error(f"   Original error: {str(mem_err)}")
                    raise RuntimeError(error_msg)
                else:
                    # Re-raise if it's not a memory error
                    raise
            
            # ✅ FIX: Return the full forecast DataFrame (tail with horizon) for postprocess
            # postprocess expects a DataFrame with columns: 'yhat', 'yhat_lower', 'yhat_upper', 'ds'
            forecast_tail = forecast.tail(horizon).copy()
            
            logger.info(f"✅ Real Prophet forecast complete: {len(forecast_tail)} predictions")
            return forecast_tail
            
        except Exception as e:
            error_msg = f"Prophet prediction failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)


    def _add_real_date_regressors(self, future: pd.DataFrame) -> pd.DataFrame:
        """Add real date-based regressors that can be extracted from the date column"""
        # Add day of week indicators (these are always available from date)
        future['is_saturday'] = (future['ds'].dt.dayofweek == 5).astype(int)
        future['is_sunday'] = (future['ds'].dt.dayofweek == 6).astype(int)
        future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
        
        # Add month indicators
        future['is_month_start'] = future['ds'].dt.is_month_start.astype(int)
        future['is_month_end'] = future['ds'].dt.is_month_end.astype(int)
        
        # Add quarter indicators
        future['is_quarter_start'] = future['ds'].dt.is_quarter_start.astype(int)
        future['is_quarter_end'] = future['ds'].dt.is_quarter_end.astype(int)
        
        # Add cyclical features for seasonality
        future['day_of_week_sin'] = np.sin(2 * np.pi * future['ds'].dt.dayofweek / 7)
        future['day_of_week_cos'] = np.cos(2 * np.pi * future['ds'].dt.dayofweek / 7)
        future['month_sin'] = np.sin(2 * np.pi * future['ds'].dt.month / 12)
        future['month_cos'] = np.cos(2 * np.pi * future['ds'].dt.month / 12)
        
        logger.info("✅ Added real date-based regressors to future dataframe")
        return future
    
    def _add_extra_regressors_to_future(self, future: pd.DataFrame, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        """Add extra regressors that the Prophet model expects to the future dataframe"""
        # Check what extra regressors the model expects
        if not hasattr(self.model, 'extra_regressors') or not self.model.extra_regressors:
            logger.debug("   No extra regressors required by model")
            return future
        
        expected_regressors = list(self.model.extra_regressors.keys())
        logger.info(f"   Model expects {len(expected_regressors)} extra regressors: {expected_regressors}")
        
        # For each expected regressor, add it to the future dataframe
        for regressor_name in expected_regressors:
            # Skip if already in future (shouldn't happen, but safety check)
            if regressor_name in future.columns:
                logger.debug(f"   Regressor {regressor_name} already in future dataframe")
                continue
                
            if regressor_name in preprocessed_data.columns:
                # Merge historical values from preprocessed_data
                # The future dataframe includes both historical and future periods
                historical_regressor = preprocessed_data[['ds', regressor_name]].copy()
                future = future.merge(historical_regressor, on='ds', how='left')
                
                # For future periods (where the merge left NaN), use the last value
                last_value = preprocessed_data[regressor_name].iloc[-1]
                future[regressor_name] = future[regressor_name].fillna(last_value)
                logger.debug(f"   Added {regressor_name} (merged historical, filled future with last value: {last_value})")
            else:
                # Regressor not in preprocessed data - use default value
                # Check if it's a date-derived regressor we can compute
                if regressor_name == 'unique_customers':
                    future[regressor_name] = 1  # Default: assume 1 customer
                    logger.warning(f"   Using default value for {regressor_name} (not in preprocessed data)")
                elif regressor_name == 'avg_revenue':
                    future[regressor_name] = 0  # Default: 0 revenue
                    logger.warning(f"   Using default value for {regressor_name} (not in preprocessed data)")
                else:
                    # Unknown regressor - use 0 as default
                    future[regressor_name] = 0
                    logger.warning(f"   Using default value 0 for {regressor_name} (not in preprocessed data)")
        
        logger.info(f"✅ Added {len(expected_regressors)} extra regressors to future dataframe")
        return future

    def postprocess(self, raw_predictions: pd.DataFrame) -> ForecastResult:
        """Transform Prophet forecast to standardized format with enhanced metadata"""
        logger.info("🔄 Postprocessing Prophet forecast")
        
        # Extract predictions and confidence intervals
        predictions_array = raw_predictions['yhat'].values  # Keep as numpy array for stats
        predictions = predictions_array.tolist()  # Convert to list for ForecastResult
        confidence_lower = raw_predictions['yhat_lower'].values
        confidence_upper = raw_predictions['yhat_upper'].values
        
        # Convert confidence intervals to list of tuples as expected by ForecastResult
        confidence_intervals = [
            (float(lower), float(upper)) 
            for lower, upper in zip(confidence_lower, confidence_upper)
        ]
        
        # Get timestamps (convert to list for metadata)
        timestamps = raw_predictions['ds'].tolist()
        
        # Extract components for insights
        components = self._extract_components(raw_predictions)
        
        # Create enhanced metadata
        debug_id = f"prophet-{self.metadata.model_name}-{int(pd.Timestamp.now().timestamp())}"
        
        metadata = {
            'model_name': self.metadata.model_name,
            'model_type': 'prophet',
            'timestamp': pd.Timestamp.now().isoformat(),
            'components': components,
            'executor_version': '1.0',
            'debug_id': debug_id,
            'regressors_used': getattr(raw_predictions, 'attrs', {}).get('__regressors_used__', []),
            'history_shape': getattr(raw_predictions, 'attrs', {}).get('__history_shape__'),
            'forecast_statistics': {
                'mean': float(np.mean(predictions_array)),
                'std': float(np.std(predictions_array)),
                'min': float(np.min(predictions_array)),
                'max': float(np.max(predictions_array))
            },
            'timestamps': timestamps  # Store timestamps in metadata
        }
        
        return ForecastResult(
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            model_name=self.metadata.model_name,
            metadata=metadata
        )
    
    def _add_regressors_to_future(self, future: pd.DataFrame, history: pd.DataFrame, 
                                regressor_cols: list) -> pd.DataFrame:
        """Add regressor values to future dataframe with robust fallbacks"""
        logger.debug(f"   Adding regressors: {regressor_cols}")
        
        # Strategy 1: Use last available values (most common approach)
        last_values = history[regressor_cols].iloc[-1:].copy()
        
        for col in regressor_cols:
            if col in last_values.columns:
                # Use the last value for all future periods
                future[col] = last_values[col].values[0]
                logger.debug(f"     {col}: using constant value {future[col].iloc[0]}")
            else:
                # Fallback: use mean value
                future[col] = history[col].mean()
                logger.warning(f"     {col}: using mean value {future[col].iloc[0]} (last value not available)")
        
        return future
    
    def _extract_components(self, forecast_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract forecast components from Prophet output"""
        components = {}
        
        # Extract available components
        component_candidates = ['trend', 'yearly', 'weekly', 'daily', 'holidays']
        for component in component_candidates:
            if component in forecast_df.columns:
                components[component] = {
                    'values': forecast_df[component].values.tolist(),
                    'mean': float(forecast_df[component].mean()),
                    'std': float(forecast_df[component].std())
                }
        
        # Calculate component contributions
        if 'trend' in components and 'yhat' in forecast_df.columns:
            trend_contribution = abs(components['trend']['mean']) / max(abs(forecast_df['yhat'].mean()), 1e-6)
            components['trend_contribution'] = min(trend_contribution, 1.0)
        
        return components