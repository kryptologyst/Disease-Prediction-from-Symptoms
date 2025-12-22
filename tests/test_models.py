"""Tests for the disease prediction system."""

import pytest
import numpy as np
from unittest.mock import patch

from src.data import SymptomDataset
from src.models import SymptomPredictor, LogisticRegressionModel, XGBoostModel
from src.evaluation import ClinicalMetrics
from src.utils import set_seed, get_device


class TestSymptomDataset:
    """Test cases for SymptomDataset."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = SymptomDataset(synthetic_size=100, random_state=42)
        assert dataset.synthetic_size == 100
        assert dataset.random_state == 42
        assert dataset.include_metadata is True
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        dataset = SymptomDataset(synthetic_size=50, random_state=42)
        dataset.load_data()
        
        assert dataset.data is not None
        assert len(dataset.data) == 50
        assert 'symptoms' in dataset.data.columns
        assert 'disease' in dataset.data.columns
    
    def test_data_splitting(self):
        """Test train-test splitting."""
        dataset = SymptomDataset(synthetic_size=100, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = dataset.split_data()
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_feature_encoding(self):
        """Test feature encoding."""
        dataset = SymptomDataset(synthetic_size=50, random_state=42)
        dataset.load_data()
        X, y = dataset.get_features()
        
        assert X.shape[0] == 50
        assert len(y) == 50
        assert X.shape[1] > 0  # Should have encoded features
    
    def test_disease_statistics(self):
        """Test disease statistics calculation."""
        dataset = SymptomDataset(synthetic_size=100, random_state=42)
        dataset.load_data()
        stats = dataset.get_disease_statistics()
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
        for disease, disease_stats in stats.items():
            assert 'count' in disease_stats
            assert 'percentage' in disease_stats
            assert 'avg_symptoms' in disease_stats


class TestModels:
    """Test cases for ML models."""
    
    def test_logistic_regression(self):
        """Test logistic regression model."""
        # Create dummy data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        probabilities = model.predict_proba(X[:10])
        
        assert len(predictions) == 10
        assert probabilities.shape == (10, 3)
        assert model.is_fitted is True
    
    def test_xgboost_model(self):
        """Test XGBoost model."""
        # Create dummy data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        
        model = XGBoostModel(random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        probabilities = model.predict_proba(X[:10])
        
        assert len(predictions) == 10
        assert probabilities.shape == (10, 3)
        assert model.is_fitted is True
    
    def test_symptom_predictor(self):
        """Test SymptomPredictor wrapper."""
        dataset = SymptomDataset(synthetic_size=100, random_state=42)
        
        predictor = SymptomPredictor(model_type="logistic", random_state=42)
        predictor.fit(dataset)
        
        # Test prediction
        symptoms = ["fever", "cough"]
        prediction = predictor.predict(symptoms)
        probabilities = predictor.predict_proba(symptoms)
        
        assert isinstance(prediction, str)
        assert isinstance(probabilities, dict)
        assert len(probabilities) > 0


class TestEvaluation:
    """Test cases for evaluation metrics."""
    
    def test_clinical_metrics(self):
        """Test clinical metrics calculation."""
        # Create dummy data
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2])
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.9, 0.05, 0.05],
            [0.1, 0.1, 0.8]
        ])
        
        class_names = ["Disease_A", "Disease_B", "Disease_C"]
        metrics_calc = ClinicalMetrics(class_names)
        
        metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        assert 'macro_auc' in metrics
        assert 'macro_auprc' in metrics
        assert 'per_class' in metrics
        assert 'calibration_error' in metrics
        
        assert isinstance(metrics['accuracy'], float)
        assert isinstance(metrics['macro_auc'], float)
        assert isinstance(metrics['per_class'], dict)


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test numpy random
        np.random.seed(42)
        val1 = np.random.random()
        
        set_seed(42)
        val2 = np.random.random()
        
        assert val1 == val2
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device is not None
        assert hasattr(device, 'type')


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline."""
        # Create dataset
        dataset = SymptomDataset(synthetic_size=200, random_state=42)
        
        # Train model
        predictor = SymptomPredictor(model_type="logistic", random_state=42)
        predictor.fit(dataset)
        
        # Test prediction
        symptoms = ["fever", "cough", "sore throat"]
        prediction = predictor.predict(symptoms)
        probabilities = predictor.predict_proba(symptoms)
        
        # Verify results
        assert isinstance(prediction, str)
        assert prediction in dataset.disease_names
        assert isinstance(probabilities, dict)
        assert len(probabilities) == len(dataset.disease_names)
        
        # Check probability sum
        prob_sum = sum(probabilities.values())
        assert abs(prob_sum - 1.0) < 1e-6
    
    def test_model_comparison(self):
        """Test comparing different models."""
        dataset = SymptomDataset(synthetic_size=100, random_state=42)
        
        models = ["logistic", "xgboost"]
        results = {}
        
        for model_type in models:
            predictor = SymptomPredictor(model_type=model_type, random_state=42)
            predictor.fit(dataset)
            
            # Evaluate
            metrics = predictor.evaluate(dataset)
            results[model_type] = metrics
        
        # Verify results
        assert len(results) == 2
        assert "logistic" in results
        assert "xgboost" in results
        
        for model_type, metrics in results.items():
            assert 'accuracy' in metrics
            assert 'average_precision' in metrics
            assert isinstance(metrics['accuracy'], float)
            assert isinstance(metrics['average_precision'], float)


if __name__ == "__main__":
    pytest.main([__file__])
