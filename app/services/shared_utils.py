# app/services/shared_utils.py

"""
SHARED UTILITIES MODULE
Centralized utility functions used across multiple services
Eliminates code duplication and ensures consistency
"""

import json
import logging
from typing import Dict, List, Any, Union
import ast

logger = logging.getLogger(__name__)


# ============================================================================
# DATA PARSING UTILITIES
# ============================================================================

def parse_features(features_data: Union[str, List, Dict]) -> List[str]:
    """
    Parse features from various formats (string, list, JSON, etc.)
    
    Used by:
    - supply_chain_service.py → _parse_required_features()
    - rule_engine_service.py → _parse_required_features()
    - rule_engine.py → (could use this)
    
    Args:
        features_data: Features in various formats
        
    Returns:
        List of feature names
    """
    if isinstance(features_data, list):
        return features_data
    
    elif isinstance(features_data, str):
        # Try JSON parsing first
        try:
            return json.loads(features_data)
        except json.JSONDecodeError:
            pass
        
        # Try Python literal eval
        try:
            result = ast.literal_eval(features_data)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass
        
        # Handle bracketed list format: "[feature1, feature2]"
        if features_data.startswith('[') and features_data.endswith(']'):
            try:
                return eval(features_data)
            except:
                pass
        
        # Last resort: split by comma
        if ',' in features_data:
            return [f.strip() for f in features_data.split(',') if f.strip()]
        
        # Single feature as string
        return [features_data.strip()] if features_data.strip() else []
    
    elif isinstance(features_data, dict):
        # Extract from dict if it has a 'features' key
        if 'features' in features_data:
            return parse_features(features_data['features'])
        # Otherwise return empty
        return []
    
    else:
        return []


def safe_json_load(json_data: Union[str, dict, None]) -> Dict[str, Any]:
    """
    Safely load JSON data with comprehensive error handling
    
    Used by:
    - rule_engine_service.py → _safe_json_load()
    - (potentially others)
    
    Args:
        json_data: JSON string, dict, or None
        
    Returns:
        Dictionary (empty if parsing fails)
    """
    if json_data is None:
        return {}
    
    if isinstance(json_data, dict):
        return json_data
    
    if isinstance(json_data, str):
        if not json_data.strip():
            return {}
        
        try:
            return json.loads(json_data)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}")
            return {}
    
    # For any other type
    try:
        return dict(json_data)
    except:
        return {}


# ============================================================================
# DATA SANITIZATION UTILITIES
# ============================================================================

def sanitize_dataset_info(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove non-serializable objects from dataset info (DataFrames, etc.)
    
    Used by:
    - rule_engine_service.py → _sanitize_dataset_info()
    - rule_engine.py → _sanitize_dataset_info()
    
    Args:
        dataset_info: Dataset information dictionary
        
    Returns:
        Sanitized dictionary without DataFrames/non-serializable objects
    """
    sanitized = dataset_info.copy()
    
    keys_to_remove = []
    for key, value in sanitized.items():
        # Check for DataFrame objects
        if hasattr(value, '__class__') and 'DataFrame' in str(value.__class__):
            keys_to_remove.append(key)
        # Check for pandas dtypes attribute
        elif hasattr(value, 'dtypes'):
            keys_to_remove.append(key)
        # Check for numpy arrays
        elif hasattr(value, '__array__'):
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        if key in sanitized:
            logger.debug(f"Removed non-serializable key: {key}")
            del sanitized[key]
    
    return sanitized

# app/services/shared_utils.py (ADD THIS NEW FUNCTION)

def normalize_model_type(db_model_type: str) -> str:
    """
    Normalize database model types to executor-compatible types
    FIXES: prophet_time_series → prophet, lightgbm_regression → lightgbm
    """
    
    # Mapping from database types to executor types
    type_mapping = {
        # Prophet variants
        'prophet_time_series': 'prophet',
        'prophet': 'prophet',
        
        # LightGBM variants
        'lightgbm_regression': 'lightgbm',
        'lightgbm_classification': 'lightgbm',
        'lightgbm': 'lightgbm',
        
        # LSTM variants
        'lstm_sequence': 'lstm',
        'lstm': 'lstm',
        
        # XGBoost variants
        'xgboost_regression': 'xgboost',
        'xgboost': 'xgboost',
        'unknown.joblib': 'xgboost',  # Fix corrupted XGBoost entry
        
        # Fallback
        'custom': 'custom',
        'unknown': 'custom'
    }
    
    normalized = type_mapping.get(db_model_type.lower(), 'custom')
    
    if normalized != db_model_type:
        logger.info(f"🔄 Normalized model type: '{db_model_type}' → '{normalized}'")
    
    return normalized

# ============================================================================
# MODEL COMPATIBILITY UTILITIES
# ============================================================================

def calculate_model_compatibility_score(
    model: Dict[str, Any],
    dataset_info: Dict[str, Any]
) -> int:
    """
    Calculate compatibility score (0-100) between model and dataset
    
    Used by:
    - supply_chain_service.py → _calculate_compatibility_score()
    - rule_engine_service.py → _calculate_model_compatibility()
    
    Args:
        model: Model metadata dictionary
        dataset_info: Dataset information dictionary
        
    Returns:
        Compatibility score (0-100)
    """
    score = 50  # Base score
    
    model_name = model.get('model_name', '').lower()
    model_type = model.get('model_type', '').lower()
    dataset_columns = set(dataset_info.get('columns', []))
    frequency = dataset_info.get('frequency', 'none')
    row_count = dataset_info.get('row_count', 0)
    
    # Feature matching (up to 30 points)
    required_features = parse_features(model.get('required_features', []))
    if required_features:
        matched_features = len(set(required_features) & dataset_columns)
        feature_ratio = matched_features / len(required_features)
        score += int(feature_ratio * 30)
    
    # Target variable match (20 points)
    target_variable = model.get('target_variable', '')
    if target_variable and target_variable in dataset_columns:
        score += 20
    
    # Model-specific bonuses (up to 20 points)
    if 'prophet' in model_type or 'prophet' in model_name:
        if frequency in ['daily', 'weekly', 'monthly']:
            score += 15
        if row_count > 100:
            score += 5
    
    elif 'lightgbm' in model_type or 'lightgbm' in model_name:
        if len(dataset_columns) > 3:
            score += 10
        if row_count > 500:
            score += 10
    
    elif 'arima' in model_type or 'arima' in model_name:
        if frequency != 'none':
            score += 10
        if row_count > 50:
            score += 10
    
    elif 'ensemble' in model_type or 'ensemble' in model_name:
        if len(dataset_columns) > 2:
            score += 10
        if row_count > 100:
            score += 10
    
    # Data size penalties
    if row_count < 30:
        score -= 20
    elif row_count < 50:
        score -= 10
    
    # Missing data penalty
    missing_pct = dataset_info.get('missing_percentage', 0)
    if missing_pct > 0.3:
        score -= 15
    elif missing_pct > 0.1:
        score -= 5
    
    return max(0, min(100, score))


def generate_compatibility_reasons(
    model: Dict[str, Any],
    dataset_info: Dict[str, Any],
    score: int
) -> List[str]:
    """
    Generate human-readable reasons for compatibility score
    
    Args:
        model: Model metadata
        dataset_info: Dataset information
        score: Calculated compatibility score
        
    Returns:
        List of reason strings
    """
    reasons = []
    
    model_name = model.get('model_name', 'Unknown')
    model_type = model.get('model_type', 'unknown')
    dataset_columns = set(dataset_info.get('columns', []))
    frequency = dataset_info.get('frequency', 'none')
    row_count = dataset_info.get('row_count', 0)
    
    # Feature matching reasons
    required_features = parse_features(model.get('required_features', []))
    matched_features = set(required_features) & dataset_columns
    if matched_features:
        reasons.append(f"Matches {len(matched_features)}/{len(required_features)} required features")
    elif required_features:
        reasons.append(f"Missing {len(required_features) - len(matched_features)} required features")
    
    # Model-specific reasons
    if 'prophet' in model_type.lower():
        if frequency in ['daily', 'weekly', 'monthly']:
            reasons.append(f"Prophet works well with {frequency} data")
        else:
            reasons.append("Prophet prefers time-series with clear frequency")
    
    if 'lightgbm' in model_type.lower():
        if len(dataset_columns) > 3:
            reasons.append("LightGBM benefits from multiple features")
        else:
            reasons.append("LightGBM performs better with more features")
    
    # Score-based interpretation
    if score >= 80:
        reasons.append("High compatibility - strongly recommended")
    elif score >= 60:
        reasons.append("Good compatibility - recommended")
    elif score >= 40:
        reasons.append("Moderate compatibility - may work")
    else:
        reasons.append("Limited compatibility - not recommended")
    
    # Data size assessment
    if row_count >= 500:
        reasons.append("Excellent data volume")
    elif row_count >= 100:
        reasons.append("Good data volume")
    elif row_count >= 50:
        reasons.append("Adequate data volume")
    else:
        reasons.append("Limited data - results may vary")
    
    return reasons if reasons else ["Basic compatibility check"]


# ============================================================================
# TARGET VARIABLE INFERENCE
# ============================================================================

def infer_target_variable(dataset_info: Dict[str, Any]) -> str:
    """
    Intelligently infer target variable from dataset columns
    
    Used by:
    - supply_chain_service.py → _infer_target_variable()
    
    Args:
        dataset_info: Dataset information dictionary
        
    Returns:
        Inferred target variable name
    """
    columns = dataset_info.get('columns', [])
    
    if not columns:
        return 'unknown'
    
    # Priority order for target detection (lowercase for comparison)
    target_priorities = [
        'sales', 'demand', 'quantity', 'value', 
        'target', 'revenue', 'volume', 'amount'
    ]
    
    # Check exact matches (case-insensitive)
    columns_lower = [col.lower() for col in columns]
    for target in target_priorities:
        if target in columns_lower:
            # Return original column name with correct case
            idx = columns_lower.index(target)
            return columns[idx]
    
    # Check partial matches
    numeric_indicators = ['amt', 'cnt', 'count', 'total', 'qty', 'num']
    for indicator in numeric_indicators:
        for i, column in enumerate(columns):
            if indicator in column.lower():
                return column
    
    # Avoid date/time columns
    date_keywords = ['date', 'time', 'datetime', 'timestamp', 'year', 'month', 'day']
    non_date_columns = [
        col for col in columns 
        if not any(kw in col.lower() for kw in date_keywords)
    ]
    
    # Return first non-date column
    if non_date_columns:
        return non_date_columns[0]
    
    # Last resort: first column
    return columns[0]


# ============================================================================
# CACHE KEY GENERATION
# ============================================================================

def generate_cache_key(dataset_info: Dict[str, Any]) -> str:
    """
    Generate reproducible cache key from dataset info
    
    Used by:
    - supply_chain_service.py → _generate_cache_key()
    - rule_engine_service.py → _generate_dataset_hash()
    
    Args:
        dataset_info: Dataset information (without DataFrame)
        
    Returns:
        MD5 hash string
    """
    import hashlib
    
    # Sanitize first to remove DataFrames
    clean_info = sanitize_dataset_info(dataset_info)
    
    # Create deterministic string representation
    cache_components = {
        'name': clean_info.get('name', 'dataset'),
        'columns': sorted([str(c) for c in clean_info.get('columns', [])]),
        'row_count': clean_info.get('row_count', 0),
        'missing_percentage': round(float(clean_info.get('missing_percentage', 0.0)), 4),
        'frequency': clean_info.get('frequency', 'none'),
        'granularity': clean_info.get('granularity', 'none')
    }
    
    # Create stable JSON string
    cache_str = json.dumps(cache_components, sort_keys=True, default=str)
    
    # Generate MD5 hash
    return hashlib.md5(cache_str.encode()).hexdigest()


# ============================================================================
# MODEL EXTRACTION UTILITIES
# ============================================================================

def extract_model_name_from_action(action: str) -> str:
    """
    Extract model name from rule action string
    
    Used by:
    - rule_engine.py → _extract_model_name()
    
    Args:
        action: Action string from YAML rule
        
    Returns:
        Model name or empty string
    """
    import re
    
    if not action:
        return ""
    
    # Try multiple patterns
    patterns = [
        r"model_name\s*=\s*['\"]([A-Za-z0-9_]+)['\"]",  # model_name = "ModelName"
        r"['\"]([A-Z][A-Za-z0-9_]+Forecaster)['\"]",   # "ModelNameForecaster"
        r"\b([A-Z][A-Za-z0-9_]+Forecaster)\b",          # ModelNameForecaster
    ]
    
    for pattern in patterns:
        match = re.search(pattern, action, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return ""


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_dataset_basic(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Basic dataset validation (used when no YAML rules available)
    
    Used by:
    - rule_engine.py → _basic_validation()
    
    Args:
        dataset_info: Dataset metadata
        
    Returns:
        Validation result dictionary
    """
    errors = []
    warnings = []
    
    row_count = dataset_info.get('row_count', 0)
    missing_percentage = dataset_info.get('missing_percentage', 0)
    columns = dataset_info.get('columns', [])
    
    # Check data volume
    if row_count < 12:
        errors.append("Insufficient data points (minimum 12 required)")
    elif row_count < 50:
        warnings.append("Limited data points may affect model performance")
    
    # Check missing data
    if missing_percentage > 0.3:
        errors.append("Too many missing values (>30%)")
    elif missing_percentage > 0.1:
        warnings.append("Moderate missing values may affect accuracy")
    
    # Check for target variable
    if not any(col in columns for col in ['sales', 'demand', 'quantity', 'value', 'revenue']):
        warnings.append("No obvious target variable found")
    
    # Check for date column
    if not any(col in columns for col in ['date', 'datetime', 'timestamp', 'time']):
        warnings.append("No date/time column found")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'applied_rules': ['basic_validation']
    }


# ============================================================================
# DEMO & TESTING
# ============================================================================

def demo_shared_utils():
    """Demonstrate shared utilities"""
    print("🧰 SHARED UTILITIES DEMO")
    print("=" * 60)
    
    # Test parse_features
    print("\n📊 Testing parse_features():")
    test_cases = [
        ['feature1', 'feature2'],
        '["feature1", "feature2"]',
        "feature1, feature2",
        "['feature1', 'feature2']"
    ]
    for case in test_cases:
        result = parse_features(case)
        print(f"   Input: {case}")
        print(f"   Output: {result}")
    
    # Test safe_json_load
    print("\n📊 Testing safe_json_load():")
    test_json = [
        '{"key": "value"}',
        {'key': 'value'},
        None,
        'invalid json'
    ]
    for case in test_json:
        result = safe_json_load(case)
        print(f"   Input: {case} → Output: {result}")
    
    # Test sanitize_dataset_info
    print("\n📊 Testing sanitize_dataset_info():")
    dataset = {
        'name': 'test',
        'columns': ['a', 'b'],
        'row_count': 100
    }
    result = sanitize_dataset_info(dataset)
    print(f"   Sanitized: {result}")
    
    # Test compatibility scoring
    print("\n📊 Testing calculate_model_compatibility_score():")
    model = {
        'model_name': 'Prophet',
        'model_type': 'prophet',
        'required_features': ['ds', 'y'],
        'target_variable': 'y'
    }
    dataset = {
        'columns': ['date', 'y', 'sales'],
        'frequency': 'daily',
        'row_count': 200
    }
    score = calculate_model_compatibility_score(model, dataset)
    print(f"   Compatibility Score: {score}/100")
    
    reasons = generate_compatibility_reasons(model, dataset, score)
    print(f"   Reasons: {reasons}")
    
    # Test target inference
    print("\n📊 Testing infer_target_variable():")
    test_datasets = [
        {'columns': ['date', 'sales', 'price']},
        {'columns': ['timestamp', 'demand', 'quantity']},
        {'columns': ['date', 'value']}
    ]
    for ds in test_datasets:
        target = infer_target_variable(ds)
        print(f"   Columns: {ds['columns']} → Target: {target}")
    
    print("\n✅ Shared utilities demo complete!")


if __name__ == "__main__":
    demo_shared_utils()