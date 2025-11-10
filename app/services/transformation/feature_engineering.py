# app/services/transformation/feature_engineering.py

"""
Enhanced Forecast Data Preprocessor - WITH SEMANTIC TYPE CHECKING
Solves the 'y' vs 'year' ambiguity and provides explicit column mapping
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime, timedelta
from app.repositories.model_repository import ModelRepository
logger = logging.getLogger(__name__)

class ForecastDataPreprocessor:
    """
    Production-ready data preprocessor with SEMANTIC TYPE CHECKING
    to prevent ambiguous mappings like 'y' vs 'year'
    """
    
    def __init__(self):
        self.model_requirements = self._load_model_requirements()
        self.feature_specs = self._load_feature_specifications()
        self.column_semantics = self._define_column_semantics()
        
    
    def _define_column_semantics(self) -> Dict[str, Dict[str, Any]]:
        """Define semantic meaning and data types for column mapping"""
        return {
            # DATE/TIME columns - should be datetime type
            "date_columns": {
                "candidates": ["date", "ds", "workdate", "timestamp", "time", "datetime"],
                "expected_dtype": "datetime",
                "description": "Date/time index for time series"
            },
            
            # TARGET VARIABLES - should be numeric, used for prediction
            "target_columns": {
                "candidates": ["sales", "revenue", "demand", "quantity", "value", "orders", "price"],
                "expected_dtype": "numeric", 
                "description": "Target variable to predict",
                # Prophet-specific: map to 'y' but with validation
                "prophet_mapping": "y"
            },
            "XGBoost Multi-Location NumberOfPieces Forecaster": {
            "type": "xgboost",
            "required_columns": ["WorkDate", "Customer", "Location", "BusinessType", "NumberOfPieces", "TotalRevenue"],
            "target_column": "NumberOfPieces",
            "feature_engineering": "monthly_aggregation_with_lags",
            "preprocessing": "onehot_encoding",
            "frequency": "monthly",
            "column_mapping": {
                "date_columns": "WorkDate",
                "target_columns": "NumberOfPieces",
                "business_columns": ["Customer", "Location", "BusinessType"]
            }
        },
            
            # ENTITY IDENTIFIERS - categorical identifiers
            "entity_columns": {
                "candidates": ["shop_id", "store_id", "location_id", "customer_id", "entity_id"],
                "expected_dtype": "categorical",
                "description": "Entity identifiers for grouping"
            },
            
            # BUSINESS ENTITIES - categorical business concepts  
            "business_columns": {
                "candidates": ["customer", "client", "location", "city", "businesstype", "service_type"],
                "expected_dtype": "categorical",
                "description": "Business entity names"
            },
            
            # COUNT/VOLUME metrics - numeric counts
            "count_columns": {
                "candidates": ["ordercount", "order_count", "count", "volume", "transactions", "quantity"],
                "expected_dtype": "numeric",
                "description": "Count or volume metrics"
            },
            
            # AMBIGUOUS COLUMNS - require special handling
            "ambiguous_columns": {
                "y": {
                    "description": "Prophet target variable - DO NOT MAP FROM USER DATA",
                    "handling": "reserved_for_prophet_only",
                    "conflicts_with": ["year"]  # Explicit conflict definition
                },
                "year": {
                    "description": "Year extracted from date - NOT a target variable", 
                    "handling": "date_derived_feature",
                    "conflicts_with": ["y"]
                }
            }
        }
    
    def _load_model_requirements(self) -> Dict[str, Any]:
        """Load model-specific input requirements with EXPLICIT mappings"""
        return {
            "Supply_Chain_Prophet_Forecaster": {
                "type": "prophet", 
                "required_columns": ["ds", "y"],
                "required_regressors": ["is_saturday", "is_sunday"],
                "optional_regressors": ["unique_customers", "avg_revenue"],
                "feature_engineering": "light",
                "preprocessing": "none",
                "column_mapping": {
                    "date_columns": "ds",
                    "target_columns": "y"
                }
            },
            "Daily_Shop_Sales_Forecaster": {
                "type": "lightgbm",
                "required_columns": ["shop_id", "date", "sales"],
                "target_column": "sales", 
                "feature_engineering": "extensive_lags_rollings",
                "preprocessing": "categorical_encoding",
                "frequency": "daily",
                "column_mapping": {
                    "date_columns": "date",
                    "target_columns": "sales", 
                    "entity_columns": "shop_id"
                }
            },
            "Monthly_Shop_Sales_Forecaster": {
                "type": "lightgbm", 
                "required_columns": ["shop_id", "date", "sales"],
                "target_column": "sales",
                "feature_engineering": "monthly_lags", 
                "preprocessing": "categorical_encoding",
                "frequency": "monthly",
                "column_mapping": {
                    "date_columns": "date",
                    "target_columns": "sales",
                    "entity_columns": "shop_id"
                }
            },
            # ✅ SIMPLIFIED LSTM REQUIREMENTS
            "LSTM Daily TotalRevenue Forecaster": {
                "type": "lstm",
                "required_columns": ["WorkDate", "Customer", "Location", "BusinessType", "OrderCount"],
                "target_column": "TotalRevenue", 
                "feature_engineering": "sequence_preparation",
                "preprocessing": "normalization",
                "frequency": "daily",
                "column_mapping": {
                    "date_columns": "WorkDate",
                    "target_columns": "TotalRevenue",
                    "business_columns": ["Customer", "Location", "BusinessType"],
                    "numeric_columns": ["OrderCount"]
                }
            },
            # ✅ UPDATED XGBOOST REQUIREMENTS (Based on your metadata)
            "XGBoost Multi-Location NumberOfPieces Forecaster": {
                "type": "xgboost", 
                "required_columns": ["WorkDate", "Customer", "Location", "BusinessType", "NumberOfPieces", "TotalRevenue"],
                "target_column": "NumberOfPieces",
                "feature_engineering": "monthly_aggregation_with_lags",
                "preprocessing": "onehot_encoding",
                "frequency": "monthly",
                "column_mapping": {
                    "date_columns": "WorkDate",
                    "target_columns": "NumberOfPieces",
                    "business_columns": ["Customer", "Location", "BusinessType"],
                    "numeric_columns": ["TotalRevenue"]
                },
                "metadata_based": {
                    "aggregation": {
                        "level": "monthly",
                        "grouping": ["Customer", "Location", "BusinessType", "YearMonth"],
                        "methods": {
                            "NumberOfPieces": "sum",
                            "TotalRevenue": "sum"
                        }
                    },
                    "temporal_features": ["month", "quarter", "year"],
                    "lag_features": [1, 3, 6],
                    "rolling_windows": [3, 6, 12],
                    "rolling_metrics": ["mean", "std"],
                    "preprocessing": {
                        "categorical_encoding": "OneHotEncoder", 
                        "numeric_scaling": "none"
                    }
                }
            },
            # Add support for the test models used in diagnostic
            "lightgbm_demand_forecaster": {
                "type": "lightgbm",
                "required_columns": ["date", "sales"],
                "target_column": "sales",
                "feature_engineering": "basic",
                "preprocessing": "none",
                "column_mapping": {
                    "date_columns": "date",
                    "target_columns": "sales"
                }
            },
            "prophet_forecaster": {
                "type": "prophet",
                "required_columns": ["ds", "y"],
                "target_column": "y",
                "feature_engineering": "light", 
                "preprocessing": "none",
                "column_mapping": {
                    "date_columns": "ds",
                    "target_columns": "y"
                }
            }
        }

    def _load_feature_specifications(self) -> Dict[str, Any]:
        """Load detailed feature specifications for each model"""
        return {
            "Daily_Shop_Sales_Forecaster": {
                "lags": [1, 2, 3, 7, 14, 21, 30],
                "rolling_windows": [3, 7, 14, 30, 60],
                "rolling_types": ["mean", "std", "median"],
                "expanding_stats": ["mean", "std"],
                "seasonal_features": True
            },
            "Monthly_Shop_Sales_Forecaster": {
                "lags": [1, 2, 3, 6, 12],
                "rolling_windows": [3, 6, 12],
                "rolling_types": ["mean", "std"],
                "expanding_stats": ["mean", "std"],
                "seasonal_features": True
            }
        }
    

    def _apply_semantic_mapping(self, data: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        ✅ FIXED: Apply SEMANTIC mapping with proper index handling
        Prevents ValueError from index misalignment
        """
        logger.info(f"🔍 Applying semantic mapping for {model_name}")
        
        # Reset index to ensure clean assignment
        data = data.copy().reset_index(drop=True)
        
        requirements = self.model_requirements.get(model_name, {})
        mapping_rules = requirements.get("column_mapping", {})
        
        applied_mappings = {}
        
        # Process each semantic group
        for semantic_group, target_column in mapping_rules.items():
            if semantic_group not in self.column_semantics:
                continue
                
            group_info = self.column_semantics[semantic_group]
            candidates = group_info.get("candidates", [])
            expected_dtype = group_info.get("expected_dtype")
            
            # If the configured target is a list (e.g., business columns), don't assign a Series to multiple columns
            if isinstance(target_column, list):
                # Ensure listed columns exist and types are sane; no remapping needed
                missing = [col for col in target_column if col not in data.columns]
                if missing:
                    logger.warning(f"⚠️ Missing columns for '{semantic_group}': {missing} (skipping semantic remap)")
                # Optionally enforce dtype expectations for existing columns
                continue
            
            # From here, target_column is a single column name (str)
            best_candidate = self._find_best_column_candidate(data, candidates, expected_dtype)
            
            if best_candidate and best_candidate != target_column:
                # Check for conflicts with ambiguous columns
                if not self._has_column_conflict(target_column, best_candidate, data):
                    # Assign safely
                    data[target_column] = data[best_candidate].copy()
                    applied_mappings[best_candidate] = target_column
                    logger.info(f"   📋 Mapped '{best_candidate}' → '{target_column}' ({semantic_group})")
        
        # SPECIAL HANDLING FOR PROPHET 'y' COLUMN
        if model_name == "Supply_Chain_Prophet_Forecaster":
            self._handle_prophet_special_cases(data, applied_mappings)
        
        logger.info(f"✅ Applied mappings: {applied_mappings}")
        return data

    def _find_best_column_candidate(self, data: pd.DataFrame, candidates: List[str], 
                                  expected_dtype: str) -> Optional[str]:
        """Find the best column candidate with type validation"""
        available_candidates = []
        
        for col in data.columns:
            col_lower = col.lower()
            # Check if column matches any candidate (case-insensitive)
            for candidate in candidates:
                if col_lower == candidate.lower():
                    # Validate data type
                    if self._validate_column_type(data[col], expected_dtype):
                        available_candidates.append(col)
                    break
        
        if not available_candidates:
            return None
        
        # Return the first available candidate (could be enhanced with scoring)
        return available_candidates[0]
    
    def _validate_column_type(self, series: pd.Series, expected_type: str) -> bool:
        """Validate that a column matches the expected data type"""
        if expected_type == "datetime":
            return pd.api.types.is_datetime64_any_dtype(series) or \
                   pd.api.types.is_string_dtype(series)  # Could be converted
        elif expected_type == "numeric":
            return pd.api.types.is_numeric_dtype(series)
        elif expected_type == "categorical":
            return pd.api.types.is_string_dtype(series) or \
                   pd.api.types.is_categorical_dtype(series)
        return True  # No type requirement
    
    def _has_column_conflict(self, target_col: str, source_col: str, data: pd.DataFrame) -> bool:
        """Check for column mapping conflicts"""
        # Special case: prevent mapping to 'y' if 'year' exists and looks like a date component
        if target_col == "y" and source_col.lower() == "year":
            if data[source_col].dtype in [np.int64, np.float64]:
                # Check if values look like years (e.g., 2023, 2024) not revenue
                unique_vals = data[source_col].unique()
                if all(1900 <= val <= 2100 for val in unique_vals if pd.notna(val)):
                    logger.warning(f"⚠️  Conflict: '{source_col}' appears to be year values, not target variable")
                    return True
        return False
    
    def _handle_prophet_special_cases(self, data: pd.DataFrame, applied_mappings: Dict[str, str]):
        """Special handling for Prophet model to avoid 'y' ambiguity"""
        # If 'y' column exists but wasn't mapped, check if it's actually year values
        if 'y' in data.columns and 'y' not in applied_mappings.values():
            if data['y'].dtype in [np.int64, np.float64]:
                unique_vals = data['y'].unique()
                # If values look like years, rename to avoid confusion
                if all(1900 <= val <= 2100 for val in unique_vals if pd.notna(val)):
                    logger.warning("⚠️  'y' column contains year values - renaming to 'year_column'")
                    data['year_column'] = data['y']
                    data.drop('y', axis=1, inplace=True)
    
    def prepare_data_for_model(self, user_data: pd.DataFrame, model_name: str, 
                          forecast_horizon: int = None) -> Optional[pd.DataFrame]:
        """
        Main method with SEMANTIC MAPPING integration
        Now supports DYNAMIC forecast horizons
        """
        logger.info(f"🔄 Preparing data for model: {model_name}")
        
        try:
            requirements = self.model_requirements.get(model_name)
            if not requirements:
                logger.error(f"❌ Unknown model: {model_name}")
                return None
            
            # STEP 1: Apply semantic mapping
            mapped_data = self._apply_semantic_mapping(user_data, model_name)
            
            # STEP 2: Determine optimal forecast horizon if not provided
            if forecast_horizon is None:
                forecast_horizon = self._determine_optimal_horizon(mapped_data, requirements)
            
            logger.info(f"📅 Using forecast horizon: {forecast_horizon} periods")
            
            # STEP 3: Model-specific transformation with horizon
            if requirements["type"] == "prophet":
                return self._prepare_prophet_data(mapped_data, requirements, forecast_horizon)
            elif requirements["type"] == "lightgbm":
                return self._prepare_lightgbm_data(mapped_data, requirements, model_name, forecast_horizon)
            elif requirements["type"] == "lstm":  # ✅ ADD LSTM SUPPORT
                return self._prepare_lstm_data(mapped_data, requirements, model_name, forecast_horizon)
            elif requirements["type"] == "xgboost":
                return self._prepare_xgboost_data(mapped_data, requirements, model_name, forecast_horizon)
            else:
                logger.error(f"❌ Unsupported model type: {requirements['type']}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Data preparation failed for {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _find_date_column(self, data: pd.DataFrame) -> Optional[str]:
        """Find the date column in dataset"""
        date_patterns = ['date', 'ds', 'workdate', 'timestamp', 'time', 'datetime']
        
        for col in data.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in date_patterns):
                return col
            
            # Check if column can be parsed as datetime
            try:
                pd.to_datetime(data[col].head(10))
                return col
            except:
                continue
        
        return None



    def _determine_optimal_horizon(self, data: pd.DataFrame, requirements: Dict[str, Any]) -> int:
        """
        Intelligently determine optimal forecast horizon based on data characteristics
        """
        # Default horizons by frequency
        frequency = requirements.get("frequency", "daily")
        horizon_map = {
            "daily": 30,      # 30 days ≈ 1 month
            "weekly": 13,     # 13 weeks ≈ 1 quarter  
            "monthly": 12,    # 12 months = 1 year
            "quarterly": 8,   # 8 quarters = 2 years
            "yearly": 3       # 3 years
        }
        
        base_horizon = horizon_map.get(frequency, 30)
        
        # Adjust based on data length
        if len(data) > 0:
            date_col = self._find_date_column(data)
            if date_col:
                try:
                    dates = pd.to_datetime(data[date_col])
                    data_length_days = (dates.max() - dates.min()).days
                    
                    # Cap horizon at 50% of available history
                    if frequency == "daily":
                        max_reasonable = max(7, int(data_length_days * 0.5))
                        base_horizon = min(base_horizon, max_reasonable)
                    elif frequency == "monthly":
                        data_length_months = data_length_days // 30
                        max_reasonable = max(3, int(data_length_months * 0.5))
                        base_horizon = min(base_horizon, max_reasonable)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not adjust horizon based on data length: {e}")
        
        logger.info(f"📊 Determined optimal horizon: {base_horizon} for {frequency} data")
        return base_horizon


    def _prepare_lightgbm_data(self, user_data: pd.DataFrame, requirements: Dict[str, Any],
                            model_name: str, forecast_horizon: int) -> pd.DataFrame:
        """
        Prepare data for LightGBM models with horizon tracking
        """
        logger.info(f"📊 Preparing LightGBM data for {model_name} (horizon: {forecast_horizon})")
        
        # Your existing LightGBM preparation code...
        # Add horizon to the returned data or metadata
        data = user_data.copy()
        
        # Store horizon in data attributes for later use
        data.attrs['forecast_horizon'] = forecast_horizon
        data.attrs['model_frequency'] = requirements.get("frequency", "daily")
        
        return data

    def _prepare_lstm_data(self, user_data: pd.DataFrame, requirements: Dict[str, Any],
                      model_name: str, forecast_horizon: int) -> pd.DataFrame:
        """
        SIMPLIFIED LSTM data preparation - without metadata_based dependency
        """
        logger.info(f"📊 Preparing LSTM data for {model_name} (horizon: {forecast_horizon})")
        
        data = user_data.copy()
        
        # Basic validation
        required_cols = requirements.get("required_columns", [])
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.error(f"❌ Missing required columns for LSTM: {missing_cols}")
            return None
        
        # ✅ SIMPLIFIED: Extract basic date features
        if 'WorkDate' in data.columns:
            data['WorkDate'] = pd.to_datetime(data['WorkDate'])
            data['year'] = data['WorkDate'].dt.year
            data['month'] = data['WorkDate'].dt.month  
            data['day'] = data['WorkDate'].dt.day
            data['dayofweek'] = data['WorkDate'].dt.dayofweek
            data['quarter'] = data['WorkDate'].dt.quarter
        
        # ✅ FIX: Use One-Hot Encoding (OHE) for categorical features instead of Label Encoding
        # The trained model expects OHE, which creates multiple binary columns per categorical feature
        categorical_cols = ['Customer', 'Location', 'BusinessType']
        for col in categorical_cols:
            if col in data.columns:
                # One-Hot Encoding: creates one binary column per unique category
                ohe_cols = pd.get_dummies(data[col], prefix=col, dtype=int)
                # Drop original categorical column
                data = data.drop(col, axis=1)
                # Add OHE columns
                data = pd.concat([data, ohe_cols], axis=1)
                logger.info(f"   One-Hot Encoded {col}: {len(ohe_cols.columns)} columns created ({ohe_cols.columns.tolist()})")
        
        # Drop WorkDate if it exists (keep only numeric features)
        if 'WorkDate' in data.columns:
            data = data.drop('WorkDate', axis=1)
        
        # Ensure all remaining columns are numeric
        for col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    data[col] = data[col].fillna(0)
                except Exception as e:
                    logger.warning(f"   Could not convert {col} to numeric: {e}")
        
        # Store metadata
        data.attrs['forecast_horizon'] = forecast_horizon
        data.attrs['model_frequency'] = requirements.get("frequency", "daily")
        data.attrs['preprocessing_required'] = True
        data.attrs['model_type'] = 'lstm'
        
        logger.info(f"✅ LSTM data prepared: {data.shape}")
        logger.info(f"   Features: {data.columns.tolist()}")
        
        return data

    def _prepare_xgboost_data(self, user_data: pd.DataFrame, requirements: Dict[str, Any],
                         model_name: str, forecast_horizon: int) -> pd.DataFrame:
        """
        Prepare data for XGBoost models - Based on actual metadata requirements
        Requires MONTHLY AGGREGATION with lag features exactly as specified
        """
        logger.info(f"📊 Preparing XGBoost data for {model_name} (horizon: {forecast_horizon})")
        
        data = user_data.copy()
        
        # Apply semantic mapping
        mapped_data = self._apply_semantic_mapping(data, model_name)
        
        # Validate required columns
        required_cols = requirements["required_columns"]
        missing_cols = [col for col in required_cols if col not in mapped_data.columns]
        if missing_cols:
            logger.error(f"❌ Missing required columns for XGBoost: {missing_cols}")
            return None
        
        # Convert date column
        date_col = 'WorkDate'
        mapped_data[date_col] = pd.to_datetime(mapped_data[date_col])
        
        # ✅ STEP 1: AGGREGATE TO MONTHLY LEVEL (EXACTLY as per metadata)
        mapped_data['YearMonth'] = mapped_data[date_col].dt.to_period('M')
        
        # Aggregation methods from metadata
        agg_dict = {
            'NumberOfPieces': 'sum',
            'TotalRevenue': 'sum'
        }
        
        # Group by Customer + Location + BusinessType + YearMonth (as per metadata)
        monthly_data = mapped_data.groupby(
            ['Customer', 'Location', 'BusinessType', 'YearMonth']
        ).agg(agg_dict).reset_index()
        
        # Convert YearMonth back to datetime
        monthly_data['WorkDate'] = monthly_data['YearMonth'].dt.to_timestamp()
        monthly_data = monthly_data.drop('YearMonth', axis=1)
        
        # Sort by Customer, Location, BusinessType, and Date (important for lags)
        monthly_data = monthly_data.sort_values(
            ['Customer', 'Location', 'BusinessType', 'WorkDate']
        ).reset_index(drop=True)
        
        logger.info(f"   Aggregated to monthly: {len(mapped_data)} rows → {len(monthly_data)} rows")
        
        # ✅ STEP 2: EXTRACT TIME FEATURES (as per metadata)
        monthly_data['month'] = monthly_data['WorkDate'].dt.month
        monthly_data['quarter'] = monthly_data['WorkDate'].dt.quarter  
        monthly_data['year'] = monthly_data['WorkDate'].dt.year
        
        # ✅ STEP 3: CREATE LAG FEATURES (EXACTLY as specified in metadata: 1, 3, 6 months)
        target_col = 'NumberOfPieces'
        
        for lag in [1, 3, 6]:  # Exactly as per metadata
            lag_col = f'lag_{lag}'
            monthly_data[lag_col] = monthly_data.groupby(
                ['Customer', 'Location', 'BusinessType']
            )[target_col].shift(lag)
        
        # ✅ STEP 4: CREATE ROLLING STATISTICS (EXACTLY as per metadata: windows 3, 6, 12)
        for window in [3, 6, 12]:  # Exactly as per metadata
            # Rolling mean
            roll_mean_col = f'roll_mean_{window}'
            monthly_data[roll_mean_col] = monthly_data.groupby(
                ['Customer', 'Location', 'BusinessType']
            )[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # Rolling std
            roll_std_col = f'roll_std_{window}'
            monthly_data[roll_std_col] = monthly_data.groupby(
                ['Customer', 'Location', 'BusinessType']  
            )[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std().shift(1)
            )
        
        # ✅ STEP 5: FILL MISSING VALUES (from lags and rolling stats)
        numeric_cols = monthly_data.select_dtypes(include=[np.number]).columns
        monthly_data[numeric_cols] = monthly_data[numeric_cols].fillna(0)
        
        # ✅ STEP 6: DROP DATE COLUMN (keep only features for prediction)
        monthly_data = monthly_data.drop('WorkDate', axis=1)
        
        # Store metadata
        monthly_data.attrs['forecast_horizon'] = forecast_horizon
        monthly_data.attrs['model_frequency'] = 'monthly'
        monthly_data.attrs['preprocessing_required'] = True
        monthly_data.attrs['model_type'] = 'xgboost'
        monthly_data.attrs['original_columns'] = user_data.columns.tolist()
        
        logger.info(f"✅ XGBoost data prepared: {monthly_data.shape}")
        logger.info(f"   Features: {monthly_data.columns.tolist()}")
        logger.info(f"   Lag features created: lag_1, lag_3, lag_6")
        logger.info(f"   Rolling features created: 3, 6, 12 month windows")
        
        return monthly_data

    def _find_target_candidate(self, data: pd.DataFrame, requirements: Dict) -> str:
        """
        Find the target column in the dataset based on requirements and data patterns
        
        Args:
            data: Preprocessed DataFrame
            requirements: Model requirements from _load_model_requirements
            
        Returns:
            Target column name
        """
        logger.info("🎯 Finding target candidate column")
        
        # 1. Try requirements first
        target_from_req = requirements.get("target_column")
        if target_from_req and target_from_req in data.columns:
            logger.info(f"✅ Using target from requirements: {target_from_req}")
            return target_from_req
        
        # 2. Try column mapping from requirements
        column_mapping = requirements.get("column_mapping", {})
        target_from_mapping = column_mapping.get("target_columns")
        if target_from_mapping and target_from_mapping in data.columns:
            logger.info(f"✅ Using target from mapping: {target_from_mapping}")
            return target_from_mapping
        
        # 3. Auto-detect from common target names
        common_targets = ['sales', 'revenue', 'demand', 'orders', 'quantity', 
                        'y', 'target', 'value', 'numberofpieces', 'ordercount',
                        'totalrevenue']
        
        for target in common_targets:
            if target in data.columns:
                logger.info(f"✅ Auto-detected target: {target}")
                return target
            # Also check case-insensitive
            if target.lower() in [col.lower() for col in data.columns]:
                matching_col = [col for col in data.columns if col.lower() == target.lower()][0]
                logger.info(f"✅ Auto-detected target (case-insensitive): {matching_col}")
                return matching_col
        
        # 4. Use first numeric column as fallback
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            fallback = numeric_cols[0]
            logger.warning(f"⚠️ Using first numeric column as target: {fallback}")
            return fallback
        
        # 5. Ultimate fallback
        logger.error("❌ No target column found!")
        return "y"  # Prophet default

    def _prepare_prophet_data(self, user_data: pd.DataFrame, requirements: Dict[str, Any],
                     forecast_horizon: int) -> pd.DataFrame:
        """
        Prepare data for Prophet models - WITH ALL REQUIRED REGRESSORS
        """
        logger.info(f"📊 Preparing Prophet data (horizon: {forecast_horizon})")
        
        data = user_data.copy()
        
        # Apply semantic mapping
        mapped_data = self._apply_semantic_mapping(data, 'Supply_Chain_Prophet_Forecaster')
        
        # Prophet requires specific column names: ds (date) and y (target)
        date_col = self._find_date_column(mapped_data)
        target_col = self._find_target_candidate(mapped_data, requirements)
        
        if not date_col or not target_col:
            logger.error("❌ Could not find date or target column for Prophet")
            return None
        
        # Rename columns to Prophet format
        prophet_data = mapped_data[[date_col, target_col]].copy()
        prophet_data = prophet_data.rename(columns={date_col: 'ds', target_col: 'y'})
        
        # ✅ ADD ALL REQUIRED REGRESSORS that Prophet model expects
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        
        # 1. Day of week features (creates is_saturday, is_sunday, etc.)
        prophet_data['day_of_week'] = prophet_data['ds'].dt.day_name()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days:
            prophet_data[f'is_{day.lower()}'] = (prophet_data['day_of_week'] == day).astype(int)
        
        # 2. ✅ CRITICAL FIX: Add unique_customers regressor
        # Calculate unique customers per day from original data
        if 'Customer' in user_data.columns and date_col in user_data.columns:
            unique_customers = user_data.groupby(date_col)['Customer'].nunique().reset_index()
            unique_customers.columns = [date_col, 'unique_customers']
            
            # Merge with prophet data
            prophet_data = prophet_data.merge(
                unique_customers, 
                left_on='ds', 
                right_on=date_col, 
                how='left'
            )
            prophet_data['unique_customers'] = prophet_data['unique_customers'].fillna(0)
            prophet_data = prophet_data.drop(date_col, axis=1)
            logger.info("✅ Added unique_customers regressor")
        else:
            # Fallback: create dummy unique_customers
            prophet_data['unique_customers'] = 1
            logger.warning("⚠️ Using dummy unique_customers (Customer column not found)")
        
        # 3. ✅ Add avg_revenue regressor (another required regressor)
        if 'TotalRevenue' in user_data.columns and 'NumberOfPieces' in user_data.columns:
            # Calculate average revenue per piece
            daily_revenue = user_data.groupby(date_col).agg({
                'TotalRevenue': 'sum',
                'NumberOfPieces': 'sum'
            }).reset_index()
            daily_revenue['avg_revenue'] = daily_revenue['TotalRevenue'] / daily_revenue['NumberOfPieces']
            daily_revenue['avg_revenue'] = daily_revenue['avg_revenue'].replace([np.inf, -np.inf], 0).fillna(0)
            
            prophet_data = prophet_data.merge(
                daily_revenue[[date_col, 'avg_revenue']], 
                left_on='ds', 
                right_on=date_col, 
                how='left'
            )
            prophet_data['avg_revenue'] = prophet_data['avg_revenue'].fillna(0)
            prophet_data = prophet_data.drop(date_col, axis=1)
            logger.info("✅ Added avg_revenue regressor")
        else:
            prophet_data['avg_revenue'] = 0
            logger.warning("⚠️ Using dummy avg_revenue")
        
        # 4. Additional time features
        prophet_data['year'] = prophet_data['ds'].dt.year
        prophet_data['month'] = prophet_data['ds'].dt.month
        prophet_data['quarter'] = prophet_data['ds'].dt.quarter
        
        # Store metadata
        prophet_data.attrs['forecast_horizon'] = forecast_horizon
        prophet_data.attrs['model_frequency'] = requirements.get("frequency", "daily")
        prophet_data.attrs['preprocessing_required'] = False
        prophet_data.attrs['model_type'] = 'prophet'
        
        logger.info(f"✅ Prophet data prepared: {prophet_data.shape}")
        logger.info(f"   Features: {prophet_data.columns.tolist()}")
        logger.info(f"   Required regressors: unique_customers={('unique_customers' in prophet_data.columns)}, avg_revenue={('avg_revenue' in prophet_data.columns)}")
        logger.info(f"   Date column: {date_col} → ds")
        logger.info(f"   Target column: {target_col} → y")
        
        return prophet_data
    # Keep the existing feature generation methods from previous version
    def _generate_basic_time_features(self, data: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Generate basic time-based features"""
        data['year'] = data[date_col].dt.year
        data['month'] = data[date_col].dt.month
        data['day'] = data[date_col].dt.day
        data['day_of_week'] = data[date_col].dt.dayofweek
        data['quarter'] = data[date_col].dt.quarter
        data['week_of_year'] = data[date_col].dt.isocalendar().week.astype(int)
        
        # Cyclical features
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Seasonal indicators
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        data['is_month_start'] = data[date_col].dt.is_month_start.astype(int)
        data['is_month_end'] = data[date_col].dt.is_month_end.astype(int)
        data['is_q4'] = (data['quarter'] == 4).astype(int)
        data['is_year_end'] = (data['month'] == 12).astype(int)
        data['is_year_start'] = (data['month'] == 1).astype(int)
        
        return data
    
    def _generate_time_series_features(self, data: pd.DataFrame, target_col: str, 
                                     model_name: str) -> pd.DataFrame:
        """Generate REAL time-series features from historical data"""
        specs = self.feature_specs.get(model_name, {})
        
        # Generate lag features
        for lag in specs.get("lags", []):
            lag_col = f'{target_col}_lag_{lag}'
            data[lag_col] = data[target_col].shift(lag)
        
        # Generate rolling features
        for window in specs.get("rolling_windows", []):
            for roll_type in specs.get("rolling_types", []):
                if roll_type == "mean":
                    col_name = f'roll_mean_{window}'
                    data[col_name] = data[target_col].rolling(window=window, min_periods=1).mean().shift(1)
                elif roll_type == "std":
                    col_name = f'roll_std_{window}'
                    data[col_name] = data[target_col].rolling(window=window, min_periods=1).std().shift(1)
                elif roll_type == "median":
                    col_name = f'roll_median_{window}'
                    data[col_name] = data[target_col].rolling(window=window, min_periods=1).median().shift(1)
        
        return data
    
    def _generate_entity_features(self, data: pd.DataFrame, target_col: Optional[str]) -> pd.DataFrame:
        """Generate entity-level features"""
        if target_col and 'shop_id' in data.columns:
            shop_stats = data.groupby('shop_id')[target_col].agg(['mean', 'std', 'max', 'min']).reset_index()
            shop_stats.columns = ['shop_id', 'shop_avg', 'shop_std', 'shop_max', 'shop_min']
            data = data.merge(shop_stats, on='shop_id', how='left')
        return data
    
    def _finalize_features(self, data: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Final cleanup"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(0)
        data = data.replace([np.inf, -np.inf], 0)
        return data
    
    def get_semantic_mapping_info(self) -> Dict[str, Any]:
        """Get information about semantic mapping for documentation"""
        return {
            "column_semantics": self.column_semantics,
            "model_requirements": {
                model: {
                    "type": req["type"],
                    "column_mapping": req.get("column_mapping", {}),
                    "target_column": req.get("target_column", "N/A")
                }
                for model, req in self.model_requirements.items()
            },
            "important_notes": [
                "Prophet uses 'y' for target variable - mapped from 'sales', 'revenue', etc.",
                "'year' column is always treated as date component, NOT target variable",
                "Semantic mapping prevents ambiguous column assignments"
            ]
        }