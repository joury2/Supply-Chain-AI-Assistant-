# app/core/data_processor.py
# Real Data Processor Implementation

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class DataProcessor:
    """Real data processing service for model preparation"""
    
    def __init__(self):
        logger.info("✅ Data Processor initialized")
        self.supported_models = ['Prophet', 'LightGBM', 'XGBoost', 'LSTM']
    
    def prepare_data(self, data_input: Union[pd.DataFrame, Dict[str, Any]], model_name: str) -> Dict[str, Any]:
        """
        Prepare data for specific model type with real processing
        
        Args:
            data_input: Can be either:
                       - DataFrame: Direct data for processing
                       - Dict: Dataset info containing 'data' key with DataFrame
            model_name: Name of the model to prepare data for
            
        Returns:
            Processed data dictionary ready for model input
        """
        logger.info(f"📊 Preparing data for model: {model_name}")
        
        try:
            # Extract DataFrame from input
            df = self._extract_dataframe(data_input)
            
            if df is None:
                logger.warning("No DataFrame found in input, creating mock processed data")
                return self._create_mock_processed_data(data_input, model_name)
            
            # Step 1: Identify date and target columns
            date_col, target_col = self._identify_key_columns(df, data_input)
            
            if date_col is None or target_col is None:
                logger.warning(f"Could not identify date ({date_col}) or target ({target_col}) columns")
                return self._create_mock_processed_data(data_input, model_name)
            
            # Step 2: Clean data
            df_clean = self._clean_data(df, date_col, target_col)
            
            # Step 3: Handle missing values
            df_clean = self._handle_missing_values(df_clean, target_col)
            
            # Step 4: Sort by date
            df_clean = df_clean.sort_values(date_col).reset_index(drop=True)
            
            # Step 5: Model-specific preparation
            if model_name in ['Prophet']:
                processed = self._prepare_for_statistical_models(df_clean, date_col, target_col, model_name)
            elif model_name in ['LightGBM', 'XGBoost', 'LSTM']:
                processed = self._prepare_for_ml_models(df_clean, date_col, target_col, model_name)
            else:
                logger.warning(f"Unknown model type: {model_name}, using generic preparation")
                processed = self._generic_preparation(df_clean, date_col, target_col)
            
            logger.info(f"✅ Data preparation completed for {model_name}")
            return processed
            
        except Exception as e:
            logger.error(f"❌ Error in data preparation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Fallback to mock data
            return self._create_mock_processed_data(data_input, model_name)
    
    def _extract_dataframe(self, data_input: Union[pd.DataFrame, Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """Extract DataFrame from various input formats"""
        if isinstance(data_input, pd.DataFrame):
            return data_input.copy()
        elif isinstance(data_input, dict) and 'data' in data_input and isinstance(data_input['data'], pd.DataFrame):
            return data_input['data'].copy()
        elif isinstance(data_input, dict) and 'dataframe' in data_input and isinstance(data_input['dataframe'], pd.DataFrame):
            return data_input['dataframe'].copy()
        else:
            # Try to create DataFrame from dict if it contains actual data
            if isinstance(data_input, dict) and any(key in data_input for key in ['values', 'records', 'data']):
                try:
                    # Check if it's a dataset info dict with actual data arrays
                    if 'columns' in data_input and 'values' in data_input:
                        return pd.DataFrame(data_input['values'], columns=data_input['columns'])
                    elif 'records' in data_input:
                        return pd.DataFrame(data_input['records'])
                except Exception as e:
                    logger.debug(f"Could not create DataFrame from dict: {e}")
            
            logger.warning("No DataFrame could be extracted from input")
            return None

    def _identify_key_columns(self, df: pd.DataFrame, original_input: Any) -> tuple:
        """Identify date and target columns"""
        date_col = None
        target_col = None
        
        # Try to find date column
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if datetime_cols:
            date_col = datetime_cols[0]
            logger.info(f"📅 Found datetime column: {date_col}")
        else:
            # Look for common date column names
            date_candidates = ['date', 'Date', 'datetime', 'DateTime', 'timestamp', 'time', 'ds']
            for col in df.columns:
                if any(cand in col.lower() for cand in date_candidates):
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_col = col
                        logger.info(f"📅 Converted column to datetime: {date_col}")
                        break
                    except Exception as e:
                        logger.debug(f"Could not convert {col} to datetime: {e}")
                        continue
        
        # Try to find target column
        target_candidates = ['demand', 'sales', 'value', 'target', 'quantity', 'revenue', 'y', 'target_value']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for candidate in target_candidates:
            for col in df.columns:
                if candidate in col.lower() and col in numeric_cols:
                    target_col = col
                    logger.info(f"🎯 Found target column: {target_col}")
                    break
            if target_col:
                break
        
        # If still not found, use first numeric column
        if not target_col and numeric_cols:
            target_col = numeric_cols[0]
            logger.info(f"🎯 Using first numeric column as target: {target_col}")
        
        logger.info(f"📌 Final columns - Date: {date_col}, Target: {target_col}")
        return date_col, target_col
    
    def _clean_data(self, df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
        """Clean dataset"""
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        if date_col in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=[date_col])
        else:
            df_clean = df_clean.drop_duplicates()
            
        if len(df_clean) < initial_rows:
            logger.info(f"🧹 Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Remove rows with missing date or target
        if date_col in df_clean.columns and target_col in df_clean.columns:
            df_clean = df_clean.dropna(subset=[date_col, target_col])
        elif target_col in df_clean.columns:
            df_clean = df_clean.dropna(subset=[target_col])
        
        # Remove outliers (optional - using IQR method)
        if target_col in df_clean.columns and len(df_clean) > 0:
            Q1 = df_clean[target_col].quantile(0.25)
            Q3 = df_clean[target_col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Only if there's variation
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = len(df_clean[(df_clean[target_col] < lower_bound) | (df_clean[target_col] > upper_bound)])
                if outliers > 0:
                    logger.info(f"📊 Found {outliers} outliers (capped instead of removed)")
                    df_clean[target_col] = df_clean[target_col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"✅ Data cleaning completed. Rows: {len(df_clean)}")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Handle missing values in target column"""
        if target_col not in df.columns:
            return df
            
        missing_count = df[target_col].isnull().sum()
        
        if missing_count > 0:
            logger.info(f"🔧 Handling {missing_count} missing values in target column")
            # Forward fill then backward fill
            df[target_col] = df[target_col].fillna(method='ffill').fillna(method='bfill')
            
            # If still missing, use mean
            remaining_missing = df[target_col].isnull().sum()
            if remaining_missing > 0:
                mean_val = df[target_col].mean()
                df[target_col] = df[target_col].fillna(mean_val)
                logger.info(f"🔧 Filled {remaining_missing} remaining missing values with mean")
        
        return df
    
    def _prepare_for_statistical_models(self, df: pd.DataFrame, date_col: str, 
                                       target_col: str, model_name: str) -> Dict[str, Any]:
        """Prepare data for Prophet models"""
        
        # Prophet expects 'ds' (datestamp) and 'y' (value)
        prepared_df = pd.DataFrame({
            'ds': df[date_col],
            'y': df[target_col]
        })
        logger.info("📊 Prepared data for Prophet model")
        
        return {
            'status': 'processed',
            'model_ready': True,
            'model_type': 'statistical',
            'data': prepared_df,
            'features_used': [date_col, target_col],
            'row_count': len(prepared_df),
            'processing_steps': ['cleaning', 'date_sorting', 'format_conversion'],
            'date_column': date_col,
            'target_column': target_col,
            'model_name': model_name
        }
    
    def _prepare_for_ml_models(self, df: pd.DataFrame, date_col: str, 
                               target_col: str, model_name: str) -> Dict[str, Any]:
        """Prepare data for ML models (LightGBM, XGBoost, LSTM)"""
        
        # ML models need feature engineering
        df_features = df.copy()
        
        # Create time-based features
        if date_col in df_features.columns:
            df_features['year'] = df_features[date_col].dt.year
            df_features['month'] = df_features[date_col].dt.month
            df_features['day'] = df_features[date_col].dt.day
            df_features['dayofweek'] = df_features[date_col].dt.dayofweek
            df_features['quarter'] = df_features[date_col].dt.quarter
            df_features['weekofyear'] = df_features[date_col].dt.isocalendar().week
            df_features['is_weekend'] = df_features['dayofweek'].isin([5, 6]).astype(int)
        
        # Create lag features
        if target_col in df_features.columns:
            for lag in [1, 7, 30]:
                if len(df_features) > lag:
                    df_features[f'lag_{lag}'] = df_features[target_col].shift(lag)
        
        # Create rolling features
        if target_col in df_features.columns:
            for window in [7, 30]:
                if len(df_features) > window:
                    df_features[f'rolling_mean_{window}'] = df_features[target_col].rolling(window=window).mean()
                    df_features[f'rolling_std_{window}'] = df_features[target_col].rolling(window=window).std()
        
        # Drop rows with NaN from feature engineering
        initial_count = len(df_features)
        df_features = df_features.dropna()
        dropped_count = initial_count - len(df_features)
        
        if dropped_count > 0:
            logger.info(f"📊 Dropped {dropped_count} rows due to feature engineering NaN values")
        
        # Get feature columns (exclude date and target)
        exclude_cols = [date_col, target_col] if date_col in df_features.columns else [target_col]
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        logger.info(f"📊 Prepared data for {model_name} with {len(feature_cols)} features")
        
        return {
            'status': 'processed',
            'model_ready': True,
            'model_type': 'ml',
            'data': df_features,
            'features_used': feature_cols,
            'target_column': target_col,
            'date_column': date_col,
            'row_count': len(df_features),
            'processing_steps': ['cleaning', 'feature_engineering', 'lag_features', 'rolling_features'],
            'feature_count': len(feature_cols),
            'model_name': model_name
        }
    
    def _generic_preparation(self, df: pd.DataFrame, date_col: str, target_col: str) -> Dict[str, Any]:
        """Generic data preparation"""
        return {
            'status': 'processed',
            'model_ready': True,
            'model_type': 'generic',
            'data': df,
            'features_used': list(df.columns),
            'date_column': date_col,
            'target_column': target_col,
            'row_count': len(df),
            'processing_steps': ['basic_cleaning']
        }
    
    def _create_mock_processed_data(self, original_input: Any, model_name: str) -> Dict[str, Any]:
        """Create mock processed data when real processing fails"""
        logger.warning("⚠️ Using mock processed data - real processing failed")
        
        # Extract what information we can from original input
        if isinstance(original_input, dict):
            columns = original_input.get('columns', [])
            row_count = original_input.get('row_count', 0)
        else:
            columns = []
            row_count = 0
        
        return {
            'status': 'mock_processed',
            'model_ready': False,
            'model_type': 'mock',
            'features_used': columns,
            'row_count': row_count,
            'processing_steps': ['mock_processing'],
            'note': 'Mock data used - check data format and availability',
            'model_name': model_name,
            'warning': 'This is mock data - forecasting may not work properly'
        }
    
    def validate_processed_data(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that processed data is ready for modeling"""
        issues = []
        warnings = []
        
        if not processed_data.get('model_ready', False):
            issues.append("Data not marked as model ready")
        
        if processed_data.get('status') == 'mock_processed':
            issues.append("Using mock data - real processing failed")
        
        row_count = processed_data.get('row_count', 0)
        if row_count < 10:
            issues.append(f"Insufficient data rows ({row_count} < 10)")
        elif row_count < 50:
            warnings.append(f"Limited data rows ({row_count}) may affect model performance")
        
        if not processed_data.get('features_used'):
            issues.append("No features identified")
        
        if 'target_column' not in processed_data:
            issues.append("No target column specified")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'ready_for_training': len(issues) == 0,
            'row_count': row_count,
            'feature_count': len(processed_data.get('features_used', [])),
            'data_status': processed_data.get('status', 'unknown')
        }
    
    def get_processing_summary(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of data processing results"""
        return {
            'processing_status': processed_data.get('status', 'unknown'),
            'model_ready': processed_data.get('model_ready', False),
            'model_type': processed_data.get('model_type', 'unknown'),
            'rows_processed': processed_data.get('row_count', 0),
            'features_generated': len(processed_data.get('features_used', [])),
            'processing_steps': processed_data.get('processing_steps', []),
            'target_column': processed_data.get('target_column', 'unknown'),
            'date_column': processed_data.get('date_column', 'unknown'),
            'model_name': processed_data.get('model_name', 'unknown')
        }


# Test function
def test_data_processor():
    """Test the data processor with sample data"""
    processor = DataProcessor()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'sales': [100 + i * 0.5 + (i % 7) * 10 + np.random.normal(0, 5) for i in range(100)],
        'region': ['North'] * 50 + ['South'] * 50
    })
    
    print("🧪 Testing Data Processor...")
    
    # Test 1: Direct DataFrame input
    print("\n1. Testing direct DataFrame input:")
    result1 = processor.prepare_data(sample_data, 'LightGBM')
    print(f"   Status: {result1.get('status')}")
    print(f"   Model Ready: {result1.get('model_ready')}")
    print(f"   Rows: {result1.get('row_count')}")
    print(f"   Features: {len(result1.get('features_used', []))}")
    
    # Test 2: Dict input with DataFrame
    print("\n2. Testing dict input with DataFrame:")
    result2 = processor.prepare_data({'data': sample_data}, 'Prophet')
    print(f"   Status: {result2.get('status')}")
    print(f"   Model Ready: {result2.get('model_ready')}")
    
    # Test 3: Validation
    print("\n3. Testing data validation:")
    validation = processor.validate_processed_data(result1)
    print(f"   Valid: {validation.get('valid')}")
    print(f"   Issues: {validation.get('issues')}")
    print(f"   Warnings: {validation.get('warnings')}")
    
    # Test 4: Summary
    print("\n4. Testing summary:")
    summary = processor.get_processing_summary(result1)
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Data Processor tests completed!")


if __name__ == "__main__":
    test_data_processor()