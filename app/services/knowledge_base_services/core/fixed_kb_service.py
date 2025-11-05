# app/services/knowledge_base_services/core/fixed_kb_service.py
"""
Fixed Knowledge Base Service with thread-safe SQLite connections
REPLACE your existing knowledge_base_service.py methods with this
"""
from typing import Dict, Any, List, Optional
import logging

from .thread_safe_db import get_thread_safe_db

logger = logging.getLogger(__name__)


class FixedKnowledgeBaseService:
    """
    Thread-safe Knowledge Base Service
    """
    
    def __init__(self, db_path: str = "supply_chain.db"):
        self.db = get_thread_safe_db(db_path)
        logger.info(f"✅ Thread-safe KB Service initialized: {db_path}")
    
    def get_all_models(self) -> List[Dict[str, Any]]:
        """
        Get all active models (THREAD-SAFE)
        """
        try:
            query = """
                SELECT model_id, model_name, model_type, model_path,
                       required_features, optional_features, target_variable,
                       performance_metrics, best_for, not_recommended_for,
                       is_active
                FROM ML_Models
                WHERE is_active = 1
                ORDER BY model_name
            """
            
            rows = self.db.execute(query)
            
            models = []
            for row in rows:
                models.append({
                    'model_id': row[0],
                    'model_name': row[1],
                    'model_type': row[2],
                    'model_path': row[3],
                    'required_features': row[4],
                    'optional_features': row[5],
                    'target_variable': row[6],
                    'performance_metrics': row[7],
                    'best_for': row[8],
                    'not_recommended_for': row[9],
                    'is_active': bool(row[10])
                })
            
            return models
            
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []
    
    
    def get_dataset_schema(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get dataset schema (THREAD-SAFE)
        """
        try:
            query = """
                SELECT dataset_id, dataset_name, description, columns_schema,
                       frequency, min_required_rows, typical_size
                FROM Dataset_Schemas
                WHERE dataset_name = ?
            """
            
            rows = self.db.execute(query, (dataset_name,))
            
            if rows:
                row = rows[0]
                return {
                    'dataset_id': row[0],
                    'dataset_name': row[1],
                    'description': row[2],
                    'columns_schema': row[3],
                    'frequency': row[4],
                    'min_required_rows': row[5],
                    'typical_size': row[6]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting schema for '{dataset_name}': {e}")
            return None
    
    def close(self):
        """Close database connection for current thread"""
        try:
            self.db.close_all()
        except:
            pass