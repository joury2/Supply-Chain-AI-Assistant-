"""Integration tests for the full forecasting pipeline.

These tests exercise the core steps a user flows through:
1. Data upload (represented by in-memory DataFrames)
2. Metadata extraction via the `ModelRepository`
3. Rule-based model selection with the `RuleEngine`
4. Model loading and inference using `ModelInferenceService`

We cover three representative scenarios that map to real models in the knowledge base:
- Prophet aggregate daily sales
- LSTM transaction-level daily revenue
- XGBoost monthly multi-location pieces
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.repositories.model_repository import get_model_repository
from app.knowledge_base.rule_layer.rule_engine import RuleEngine
from app.services.transformation.feature_engineering import ForecastDataPreprocessor
from app.services.inference.model_inference_service import ModelInferenceService
from app.services.knowledge_base_services.core.knowledge_base_service import (
    SupplyChainService as KnowledgeBaseService,
)


@pytest.fixture(scope="module")
def pipeline_components():
    """Initialise shared services once for this module."""
    repository = get_model_repository("supply_chain.db")
    rule_engine = RuleEngine(model_repository=repository)
    knowledge_base = KnowledgeBaseService("supply_chain.db")

    inference_service = ModelInferenceService(model_storage_path="models/")
    inference_service.set_knowledge_base(knowledge_base)

    preprocessor = ForecastDataPreprocessor()

    components = {
        "repository": repository,
        "rule_engine": rule_engine,
        "knowledge_base": knowledge_base,
        "inference": inference_service,
        "preprocessor": preprocessor,
    }

    yield components

    # Cleanup after all tests in this module finish
    for model_name in inference_service.list_loaded_models():
        inference_service.unload_model(model_name)
    knowledge_base.close()


def _build_dataset_info(metadata: dict, *, name: str, data: pd.DataFrame) -> dict:
    """Combine extracted metadata with additional request context."""
    dataset_info = metadata.copy()
    dataset_info.update(
        {
            "name": name,
            "data": data,
            "row_count": len(data),
            "missing_percentage": float(metadata.get("missing_percentage", 0.0) or 0.0),
        }
    )
    return dataset_info


def _run_pipeline(components: dict, data: pd.DataFrame, *, name: str, horizon: int):
    """Execute the four major pipeline stages for a given dataset."""
    repository = components["repository"]
    rule_engine = components["rule_engine"]
    knowledge_base = components["knowledge_base"]
    inference_service = components["inference"]
    preprocessor = components["preprocessor"]

    # Metadata extraction
    metadata = repository.extract_dataset_metadata(data, uploaded_filename=f"{name}.csv")
    dataset_info = _build_dataset_info(metadata, name=name, data=data)

    # Rule validation & selection
    validation = rule_engine.validate_dataset(dataset_info)
    assert validation["valid"], f"Validation failed for {name}: {validation['errors']}"

    selection = rule_engine.select_model(dataset_info)
    model_name = selection.get("selected_model")
    assert model_name, f"Model selection failed for {name}"

    # Ensure the correct model artefact is loaded
    if not inference_service.is_model_loaded(model_name):
        loaded = inference_service.load_model_from_registry(model_name, knowledge_base)
        assert loaded, f"Failed to load model {model_name}"

    # Data preparation tailored to the selected model
    prepared = preprocessor.prepare_data_for_model(
        user_data=data, model_name=model_name, forecast_horizon=horizon
    )
    assert prepared is not None, f"Data preparation returned None for {model_name}"

    # Final inference
    forecast_result = inference_service.generate_forecast(
        model_name=model_name,
        data=prepared,
        horizon=horizon,
    )
    assert forecast_result is not None, f"Inference failed for {model_name}"

    return selection, forecast_result


def test_prophet_end_to_end_pipeline(pipeline_components):
    """End-to-end validation for the Prophet aggregate daily pipeline."""
    horizon = 30
    prophet_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=120, freq="D"),
            "sales": [1_000 + i * 5 for i in range(120)],
        }
    )

    selection, forecast = _run_pipeline(
        pipeline_components,
        prophet_df,
        name="prophet_daily_sales",
        horizon=horizon,
    )

    assert (
        selection["selected_model"] == "Supply_Chain_Prophet_Forecaster"
    ), "Expected Prophet model to be selected"
    assert len(forecast.predictions) == horizon
    assert forecast.metadata.get("model_type") == "prophet"
    assert forecast.metadata.get("model_name") == selection["selected_model"]


def test_lstm_end_to_end_pipeline(pipeline_components):
    """End-to-end validation for the LSTM transaction revenue pipeline."""
    pytest.importorskip("tensorflow", reason="TensorFlow is required for LSTM inference")

    horizon = 45
    lstm_df = pd.DataFrame(
        {
            "WorkDate": pd.date_range("2024-01-01", periods=120, freq="D"),
            "Customer": ["Walmart"] * 120,
            "Location": ["New York"] * 120,
            "BusinessType": ["Final Mile"] * 120,
            "OrderCount": [25 + (i % 10) for i in range(120)],
            "TotalRevenue": [2_500 + i * 25 for i in range(120)],
        }
    )

    selection, forecast = _run_pipeline(
        pipeline_components,
        lstm_df,
        name="lstm_daily_revenue",
        horizon=horizon,
    )

    assert (
        selection["selected_model"] == "LSTM Daily TotalRevenue Forecaster"
    ), "Expected LSTM model to be selected"
    assert len(forecast.predictions) == horizon
    assert forecast.metadata.get("model_type") == "lstm"


def test_xgboost_end_to_end_pipeline(pipeline_components):
    """End-to-end validation for the XGBoost monthly pieces pipeline."""
    pytest.importorskip("xgboost", reason="XGBoost is required to load the model artefact")

    horizon = 6
    xgb_df = pd.DataFrame(
        {
            "WorkDate": pd.date_range("2022-01-01", periods=24, freq="MS"),
            "Customer": ["Amazon"] * 24,
            "Location": ["Dubai"] * 24,
            "BusinessType": ["Middle Mile"] * 24,
            "NumberOfPieces": [500 + i * 15 for i in range(24)],
            "TotalRevenue": [5_000 + i * 120 for i in range(24)],
        }
    )

    selection, forecast = _run_pipeline(
        pipeline_components,
        xgb_df,
        name="xgboost_monthly_pieces",
        horizon=horizon,
    )

    assert (
        selection["selected_model"] == "XGBoost Multi-Location NumberOfPieces Forecaster"
    ), "Expected XGBoost model to be selected"
    assert len(forecast.predictions) == horizon
    # Depending on executor wiring this may resolve to a generic executor; still ensure model identity
    assert forecast.metadata.get("model_name") == selection["selected_model"]

