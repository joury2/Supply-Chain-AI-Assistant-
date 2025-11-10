# app/repositories/model_repository.py
"""
Model Repository - Single source of truth for model data
Breaks circular dependency by providing a shared data access layer
"""

import sqlite3
import logging
from typing import List, Dict, Any, Optional
import json
import pandas as pd
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ModelRepository:
    """
    Repository pattern for model data access
    This is the ONLY place that talks to the ML_Models table
    """
    
    def __init__(self, db_path: str = "supply_chain.db"):
        self.db_path = db_path
        self._cache = {}
        logger.info(f"✅ ModelRepository initialized: {db_path}")
    
    @contextmanager
    def get_connection(self):
        """Thread-safe connection context manager"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def get_all_active_models(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get all active models - PRIMARY DATA ACCESS METHOD
        
        Returns:
            List of model dictionaries with all fields
        """
        cache_key = 'all_active_models'
        
        if use_cache and cache_key in self._cache:
            logger.debug("📦 Returning cached models")
            return self._cache[cache_key]
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        model_id, model_name, model_type, model_path,
                        required_features, optional_features, target_variable,
                        performance_metrics, training_config, hyperparameters,
                        is_active, created_at
                    FROM ML_Models 
                    WHERE is_active = TRUE
                    ORDER BY model_name
                """)
                
                rows = cursor.fetchall()
                
                models = []
                for row in rows:
                    models.append({
                        'model_id': row[0],
                        'model_name': row[1],
                        'model_type': row[2],
                        'model_path': row[3],
                        'required_features': self._parse_json_field(row[4]),
                        'optional_features': self._parse_json_field(row[5]),
                        'target_variable': row[6],
                        'performance_metrics': self._parse_json_field(row[7]),
                        'training_config': self._parse_json_field(row[8]),
                        'hyperparameters': self._parse_json_field(row[9]),
                        'is_active': bool(row[10]),
                        'created_at': row[11]
                    })
                
                if use_cache:
                    self._cache[cache_key] = models
                
                logger.info(f"✅ Retrieved {len(models)} active models")
                return models
                
        except Exception as e:
            logger.error(f"❌ Error fetching models: {e}")
            return []
    

    def get_model_by_name(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get specific model by name"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        model_id, model_name, model_type, model_path,
                        required_features, optional_features, target_variable,
                        performance_metrics, training_config, hyperparameters,
                        is_active, created_at
                    FROM ML_Models 
                    WHERE model_name = ? AND is_active = TRUE
                """, (model_name,))
                
                row = cursor.fetchone()
                
                if not row:
                    logger.warning(f"⚠️ Model not found: {model_name}")
                    return None
                
                return {
                    'model_id': row[0],
                    'model_name': row[1],
                    'model_type': row[2],
                    'model_path': row[3],
                    'required_features': self._parse_json_field(row[4]),
                    'optional_features': self._parse_json_field(row[5]),
                    'target_variable': row[6],
                    'performance_metrics': self._parse_json_field(row[7]),
                    'training_config': self._parse_json_field(row[8]),
                    'hyperparameters': self._parse_json_field(row[9]),
                    'is_active': bool(row[10]),
                    'created_at': row[11]
                }
                
        except Exception as e:
            logger.error(f"❌ Error fetching model {model_name}: {e}")
            return None
    
    def get_model_by_id(self, model_id: int) -> Optional[Dict[str, Any]]:
        """Get specific model by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        model_id, model_name, model_type, model_path,
                        required_features, optional_features, target_variable,
                        performance_metrics, training_config, hyperparameters,
                        is_active, created_at
                    FROM ML_Models 
                    WHERE model_id = ? AND is_active = TRUE
                """, (model_id,))
                
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return {
                    'model_id': row[0],
                    'model_name': row[1],
                    'model_type': row[2],
                    'model_path': row[3],
                    'required_features': self._parse_json_field(row[4]),
                    'optional_features': self._parse_json_field(row[5]),
                    'target_variable': row[6],
                    'performance_metrics': self._parse_json_field(row[7]),
                    'training_config': self._parse_json_field(row[8]),
                    'hyperparameters': self._parse_json_field(row[9]),
                    'is_active': bool(row[10]),
                    'created_at': row[11]
                }
                
        except Exception as e:
            logger.error(f"❌ Error fetching model {model_id}: {e}")
            return None
    
    def _verify_model_exists(self, model_name: str) -> bool:
        """Verify that a model exists in the database"""
        try:
            if self:
                model_info = self.get_model_by_name(model_name)
                return model_info is not None
            
            # Fallback: Check if model name looks reasonable
            valid_model_names = [
                "LightGBM Daily Sales Forecaster",
                "LightGBM Monthly Sales Forecaster", 
                "LSTM Revenue Forecaster",
                "Prophet Order Forecaster",
                "XGBoost Multi-Location NumberOfPieces Forecaster"
            ]
            return model_name in valid_model_names
            
        except Exception as e:
            logger.warning(f"⚠️ Could not verify model existence for {model_name}: {e}")
            return False

    def _parse_json_field(self, field_value: Any) -> Any:
        """Parse JSON string fields"""
        if field_value is None:
            return None
        
        if isinstance(field_value, str):
            try:
                return json.loads(field_value)
            except json.JSONDecodeError:
                # Try eval as fallback for Python literals
                try:
                    import ast
                    return ast.literal_eval(field_value)
                except:
                    return field_value
        
        return field_value
    
    def clear_cache(self):
        """Clear repository cache"""
        self._cache.clear()
        logger.info("🧹 ModelRepository cache cleared")
    
    def get_models_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        """Get models filtered by type"""
        all_models = self.get_all_active_models()
        return [m for m in all_models if m['model_type'].lower() == model_type.lower()]
    
    def get_models_for_target(self, target_variable: str) -> List[Dict[str, Any]]:
        """Get models that predict a specific target variable"""
        all_models = self.get_all_active_models()
        return [m for m in all_models if m['target_variable'] == target_variable]

    
    def extract_dataset_metadata(self, data: pd.DataFrame, 
                             uploaded_filename: str = None) -> Dict[str, Any]:
        """
        Extract comprehensive dataset metadata for rule evaluation
        
        Args:
            data: Uploaded DataFrame
            uploaded_filename: Optional filename for additional hints
            
        Returns:
            Dictionary with all metadata needed by rule engine
        """
        import pandas as pd
        
        metadata = {
            'columns': data.columns.tolist(),
            'row_count': len(data),
            'missing_percentage': data.isnull().sum().sum() / (len(data) * len(data.columns)),
            'filename': uploaded_filename or 'unknown',
        }
        
        # ✅ EXTRACT GRANULARITY from column patterns
        columns_lower = [col.lower() for col in data.columns]
        
        if 'shop_id' in columns_lower or 'store_id' in columns_lower:
            metadata['granularity'] = 'shop_level'
        elif 'location' in columns_lower and 'customer' in columns_lower:
            metadata['granularity'] = 'transaction_level'
        elif 'location' in columns_lower:
            metadata['granularity'] = 'location_level'
        elif any(col in columns_lower for col in ['customer', 'client']):
            metadata['granularity'] = 'customer_level'
        else:
            metadata['granularity'] = 'aggregate'  # Default for Prophet-style data
        
        # ✅ EXTRACT FREQUENCY from date patterns
        date_col = self._find_date_column(data)
        if date_col:
            metadata['frequency'] = self._infer_frequency(data[date_col])
            metadata['date_column'] = date_col
        else:
            metadata['frequency'] = 'none'
            metadata['date_column'] = None
        
        # ✅ EXTRACT TARGET VARIABLE CANDIDATES
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        target_candidates = [col for col in numeric_cols 
                            if col.lower() in ['sales', 'revenue', 'totalrevenue', 
                                            'demand', 'numberofpieces', 'pieces', 
                                            'orders', 'quantity', 'y', 'value']]
        metadata['target_candidates'] = target_candidates
        
        logger.info(f"📊 Dataset metadata extracted:")
        logger.info(f"   Granularity: {metadata['granularity']}")
        logger.info(f"   Frequency: {metadata['frequency']}")
        logger.info(f"   Target candidates: {metadata['target_candidates']}")
        
        return metadata

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


    def _infer_frequency(self, date_series: pd.Series) -> str:
        """Infer time series frequency from date column"""
        try:
            dates = pd.to_datetime(date_series)
            dates = dates.sort_values()
            
            # Calculate most common difference between consecutive dates
            diffs = dates.diff().dropna()
            most_common_diff = diffs.mode()[0] if len(diffs) > 0 else pd.Timedelta(days=1)
            
            days = most_common_diff.days
            
            if days <= 1:
                return 'daily'
            elif 6 <= days <= 8:
                return 'weekly'
            elif 28 <= days <= 32:
                return 'monthly'
            elif 88 <= days <= 95:
                return 'quarterly'
            elif days >= 360:
                return 'yearly'
            else:
                return 'irregular'
                
        except Exception as e:
            logger.warning(f"Could not infer frequency: {e}")
            return 'unknown'

# Singleton instance for global access
_repository_instance = None

def get_model_repository(db_path: str = "supply_chain.db") -> ModelRepository:
    """Get or create singleton repository instance"""
    global _repository_instance
    if _repository_instance is None:
        _repository_instance = ModelRepository(db_path)
    return _repository_instance