"""Machine learning models for symptom-based disease prediction."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all prediction models."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the model.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to training data.
        
        Args:
            X: Training features.
            y: Training labels.
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features to predict.
            
        Returns:
            Predicted labels.
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Features to predict.
            
        Returns:
            Predicted probabilities.
        """
        pass
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available.
        
        Returns:
            Feature importance array or None.
        """
        return None


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model for disease prediction."""
    
    def __init__(self, max_iter: int = 1000, random_state: int = 42):
        """Initialize logistic regression model.
        
        Args:
            max_iter: Maximum number of iterations.
            random_state: Random seed.
        """
        super().__init__(random_state)
        self.max_iter = max_iter
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit logistic regression model."""
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            multi_class='ovr'
        )
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Logistic Regression model fitted")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)


class XGBoostModel(BaseModel):
    """XGBoost model for disease prediction."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        """Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate.
            random_state: Random seed.
        """
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit XGBoost model."""
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric='mlogloss'
        )
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("XGBoost model fitted")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.feature_importances_


class LightGBMModel(BaseModel):
    """LightGBM model for disease prediction."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        """Initialize LightGBM model.
        
        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate.
            random_state: Random seed.
        """
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit LightGBM model."""
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=-1
        )
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("LightGBM model fitted")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.feature_importances_


class DeepTabularModel(BaseModel):
    """Deep neural network for tabular data."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        device: Optional[torch.device] = None
    ):
        """Initialize deep tabular model.
        
        Args:
            input_dim: Input feature dimension.
            num_classes: Number of output classes.
            hidden_dims: List of hidden layer dimensions.
            dropout_rate: Dropout rate.
            learning_rate: Learning rate.
            batch_size: Batch size.
            epochs: Number of training epochs.
            device: PyTorch device.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or torch.device("cpu")
        
        # Initialize model
        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def _build_model(self) -> nn.Module:
        """Build the neural network architecture."""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.num_classes))
        
        return nn.Sequential(*layers).to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the deep model."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.is_fitted = True
        logger.info("Deep tabular model fitted")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
            return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()


class SymptomPredictor:
    """Main predictor class that manages multiple models."""
    
    def __init__(
        self,
        model_type: str = "logistic",
        random_state: int = 42,
        device: Optional[torch.device] = None
    ):
        """Initialize the symptom predictor.
        
        Args:
            model_type: Type of model to use.
            random_state: Random seed.
            device: PyTorch device for deep models.
        """
        self.model_type = model_type
        self.random_state = random_state
        self.device = device
        self.model: Optional[BaseModel] = None
        self.symptom_names: Optional[List[str]] = None
        self.disease_names: Optional[List[str]] = None
        
    def _create_model(self, input_dim: int, num_classes: int) -> BaseModel:
        """Create model instance based on type."""
        if self.model_type == "logistic":
            return LogisticRegressionModel(random_state=self.random_state)
        elif self.model_type == "xgboost":
            return XGBoostModel(random_state=self.random_state)
        elif self.model_type == "lightgbm":
            return LightGBMModel(random_state=self.random_state)
        elif self.model_type == "deep":
            return DeepTabularModel(
                input_dim=input_dim,
                num_classes=num_classes,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, dataset) -> None:
        """Fit the model using the dataset.
        
        Args:
            dataset: SymptomDataset instance.
        """
        # Get training data
        X_train, X_test, y_train, y_test = dataset.split_data()
        
        # Store feature names and dataset reference
        self.symptom_names = dataset.symptom_names
        self.disease_names = dataset.disease_names
        self.dataset = dataset
        
        # Create and fit model
        self.model = self._create_model(X_train.shape[1], len(self.disease_names))
        self.model.fit(X_train, y_train)
        
        logger.info(f"Model {self.model_type} fitted successfully")
    
    def predict(self, symptoms: List[str]) -> str:
        """Predict disease from symptoms.
        
        Args:
            symptoms: List of symptoms.
            
        Returns:
            Predicted disease name.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Use the same encoder that was used during training
        # Transform input symptoms using the fitted encoder
        X = self.dataset.symptom_encoder.transform([symptoms])
        
        # Add metadata features if they were used during training
        if self.dataset.include_metadata:
            # Create dummy metadata (age=50, gender="Male", severity="Moderate")
            import numpy as np
            import pandas as pd
            
            # Age (normalized to match training data)
            age_normalized = np.array([0.0])  # Mean-centered age
            metadata_features = [age_normalized.reshape(-1, 1)]
            
            # Gender (one-hot encoded)
            gender_encoded = np.array([[1, 0]])  # Male
            metadata_features.append(gender_encoded)
            
            # Severity (ordinal encoded)
            severity_encoded = np.array([[1]])  # Moderate
            metadata_features.append(severity_encoded)
            
            # Combine all features
            metadata_array = np.hstack(metadata_features)
            X = np.hstack([X, metadata_array])
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        return self.disease_names[prediction]
    
    def predict_proba(self, symptoms: List[str]) -> Dict[str, float]:
        """Predict disease probabilities from symptoms.
        
        Args:
            symptoms: List of symptoms.
            
        Returns:
            Dictionary mapping disease names to probabilities.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Use the same encoder that was used during training
        # Transform input symptoms using the fitted encoder
        X = self.dataset.symptom_encoder.transform([symptoms])
        
        # Add metadata features if they were used during training
        if self.dataset.include_metadata:
            # Create dummy metadata (age=50, gender="Male", severity="Moderate")
            import numpy as np
            
            # Age (normalized to match training data)
            age_normalized = np.array([0.0])  # Mean-centered age
            metadata_features = [age_normalized.reshape(-1, 1)]
            
            # Gender (one-hot encoded)
            gender_encoded = np.array([[1, 0]])  # Male
            metadata_features.append(gender_encoded)
            
            # Severity (ordinal encoded)
            severity_encoded = np.array([[1]])  # Moderate
            metadata_features.append(severity_encoded)
            
            # Combine all features
            metadata_array = np.hstack(metadata_features)
            X = np.hstack([X, metadata_array])
        
        # Get probabilities
        probabilities = self.model.predict_proba(X)[0]
        
        return dict(zip(self.disease_names, probabilities))
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available.
        
        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.get_feature_importance()
        if importance is None:
            return None
        
        # Map to symptom names (assuming first features are symptoms)
        if len(importance) <= len(self.symptom_names):
            return dict(zip(self.symptom_names, importance))
        else:
            # Include metadata features
            all_features = self.symptom_names + ["age", "gender_male", "gender_female", "severity"]
            return dict(zip(all_features, importance))
    
    def evaluate(self, dataset) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            dataset: SymptomDataset instance.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        # Get test data
        X_test = dataset.X_test
        y_test = dataset.y_test
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate average precision for each class
        from sklearn.metrics import average_precision_score
        avg_precision = average_precision_score(
            y_test, y_proba, average='macro'
        )
        
        return {
            "accuracy": accuracy,
            "average_precision": avg_precision
        }
