# app/services/inference/__init__.py
"""
Inference package - automatically registers all executors
"""
from typing import Any, Optional, Dict

# Import all executors to ensure they register themselves
from app.services.inference.executors.lightgbm_executor import LightGBMExecutor
from app.services.inference.executors.prophet_executor import ProphetExecutor
from app.services.inference.executors.lstm_executor import LSTMExecutor
from app.services.inference.executors.generic_executor import GenericExecutor

__all__ = [
    'LightGBMExecutor',
    'ProphetExecutor', 
    'LSTMExecutor',
    'GenericExecutor'
]