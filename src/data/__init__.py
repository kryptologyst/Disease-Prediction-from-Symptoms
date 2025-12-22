"""Data handling and preprocessing for symptom-based disease prediction."""

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class SymptomDataset:
    """Dataset class for symptom-based disease prediction.
    
    This class handles synthetic data generation, preprocessing, and splitting
    for symptom-disease prediction tasks.
    """
    
    def __init__(
        self,
        synthetic_size: int = 1000,
        test_size: float = 0.2,
        random_state: int = 42,
        include_metadata: bool = True
    ):
        """Initialize the dataset.
        
        Args:
            synthetic_size: Number of synthetic samples to generate.
            test_size: Fraction of data to use for testing.
            random_state: Random seed for reproducibility.
            include_metadata: Whether to include patient metadata features.
        """
        self.synthetic_size = synthetic_size
        self.test_size = test_size
        self.random_state = random_state
        self.include_metadata = include_metadata
        
        # Initialize encoders
        self.symptom_encoder = MultiLabelBinarizer()
        self.disease_encoder = LabelEncoder()
        
        # Data storage
        self.data: pd.DataFrame = None
        self.X_train: np.ndarray = None
        self.X_test: np.ndarray = None
        self.y_train: np.ndarray = None
        self.y_test: np.ndarray = None
        
        # Feature names
        self.symptom_names: List[str] = None
        self.disease_names: List[str] = None
        
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic symptom-disease data.
        
        Returns:
            DataFrame with synthetic symptom-disease mappings.
        """
        np.random.seed(self.random_state)
        
        # Define symptom-disease mappings with realistic probabilities
        disease_symptoms = {
            "Flu": {
                "symptoms": ["fever", "cough", "sore throat", "fatigue", "body aches", "headache"],
                "probabilities": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
            },
            "Migraine": {
                "symptoms": ["headache", "nausea", "dizziness", "light sensitivity", "fatigue"],
                "probabilities": [0.95, 0.6, 0.5, 0.7, 0.4]
            },
            "Heart Attack": {
                "symptoms": ["chest pain", "shortness of breath", "sweating", "nausea", "dizziness"],
                "probabilities": [0.9, 0.8, 0.7, 0.5, 0.4]
            },
            "Food Poisoning": {
                "symptoms": ["abdominal pain", "diarrhea", "vomiting", "nausea", "fever"],
                "probabilities": [0.9, 0.8, 0.7, 0.6, 0.3]
            },
            "Diabetes": {
                "symptoms": ["fatigue", "weight loss", "frequent urination", "thirst", "blurred vision"],
                "probabilities": [0.7, 0.6, 0.8, 0.7, 0.4]
            },
            "Allergy": {
                "symptoms": ["itching", "rash", "swelling", "sneezing", "runny nose"],
                "probabilities": [0.8, 0.7, 0.6, 0.5, 0.4]
            },
            "Arthritis": {
                "symptoms": ["joint pain", "stiffness", "swelling", "fatigue", "limited mobility"],
                "probabilities": [0.9, 0.8, 0.6, 0.5, 0.4]
            },
            "Spinal Disc Problem": {
                "symptoms": ["back pain", "numbness", "tingling", "muscle weakness", "limited mobility"],
                "probabilities": [0.9, 0.6, 0.7, 0.5, 0.4]
            },
            "Pneumonia": {
                "symptoms": ["cough", "fever", "shortness of breath", "chest pain", "fatigue"],
                "probabilities": [0.9, 0.8, 0.7, 0.5, 0.6]
            },
            "Hypertension": {
                "symptoms": ["headache", "dizziness", "fatigue", "chest pain", "shortness of breath"],
                "probabilities": [0.4, 0.3, 0.5, 0.3, 0.2]
            }
        }
        
        # Collect all unique symptoms
        all_symptoms = set()
        for disease_info in disease_symptoms.values():
            all_symptoms.update(disease_info["symptoms"])
        all_symptoms = sorted(list(all_symptoms))
        
        # Generate synthetic data
        data = []
        diseases = list(disease_symptoms.keys())
        
        for _ in range(self.synthetic_size):
            # Randomly select a disease
            disease = np.random.choice(diseases)
            disease_info = disease_symptoms[disease]
            
            # Generate symptoms based on probabilities
            symptoms = []
            for symptom, prob in zip(disease_info["symptoms"], disease_info["probabilities"]):
                if np.random.random() < prob:
                    symptoms.append(symptom)
            
            # Add some random noise symptoms (rare)
            for symptom in all_symptoms:
                if symptom not in disease_info["symptoms"] and np.random.random() < 0.05:
                    symptoms.append(symptom)
            
            # Ensure at least one symptom
            if not symptoms:
                symptoms = [np.random.choice(disease_info["symptoms"])]
            
            # Generate metadata if requested
            metadata = {}
            if self.include_metadata:
                metadata = {
                    "age": np.random.randint(18, 80),
                    "gender": np.random.choice(["Male", "Female"]),
                    "severity": np.random.choice(["Mild", "Moderate", "Severe"])
                }
            
            data.append({
                "symptoms": symptoms,
                "disease": disease,
                **metadata
            })
        
        return pd.DataFrame(data)
    
    def load_data(self) -> None:
        """Load and preprocess the dataset."""
        logger.info(f"Generating synthetic dataset with {self.synthetic_size} samples")
        
        # Generate synthetic data
        self.data = self._generate_synthetic_data()
        
        # Extract symptoms and diseases
        symptoms_list = self.data["symptoms"].tolist()
        diseases = self.data["disease"].values
        
        # Fit encoders
        self.symptom_encoder.fit(symptoms_list)
        self.disease_encoder.fit(diseases)
        
        # Get feature names
        self.symptom_names = self.symptom_encoder.classes_.tolist()
        self.disease_names = self.disease_encoder.classes_.tolist()
        
        logger.info(f"Found {len(self.symptom_names)} unique symptoms")
        logger.info(f"Found {len(self.disease_names)} unique diseases")
        
    def get_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get encoded features and labels.
        
        Returns:
            Tuple of (features, labels).
        """
        if self.data is None:
            self.load_data()
        
        # Encode symptoms
        symptoms_list = self.data["symptoms"].tolist()
        X = self.symptom_encoder.transform(symptoms_list)
        
        # Encode diseases
        y = self.disease_encoder.transform(self.data["disease"])
        
        # Add metadata features if requested
        if self.include_metadata:
            metadata_features = []
            
            # Age (normalized)
            age = self.data["age"].values
            age_normalized = (age - age.mean()) / age.std()
            metadata_features.append(age_normalized.reshape(-1, 1))
            
            # Gender (one-hot encoded)
            gender_encoded = pd.get_dummies(self.data["gender"]).values
            metadata_features.append(gender_encoded)
            
            # Severity (ordinal encoded)
            severity_map = {"Mild": 0, "Moderate": 1, "Severe": 2}
            severity_encoded = self.data["severity"].map(severity_map).values
            metadata_features.append(severity_encoded.reshape(-1, 1))
            
            # Combine all features
            metadata_array = np.hstack(metadata_features)
            X = np.hstack([X, metadata_array])
        
        return X, y
    
    def split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        X, y = self.get_features()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Training set size: {len(self.X_train)}")
        logger.info(f"Test set size: {len(self.X_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_sample_symptoms(self, disease: str, n_samples: int = 5) -> List[List[str]]:
        """Get sample symptom combinations for a given disease.
        
        Args:
            disease: Disease name.
            n_samples: Number of samples to return.
            
        Returns:
            List of symptom combinations.
        """
        if self.data is None:
            self.load_data()
        
        disease_data = self.data[self.data["disease"] == disease]
        if len(disease_data) == 0:
            return []
        
        # Sample symptom combinations
        sample_indices = np.random.choice(
            len(disease_data), 
            size=min(n_samples, len(disease_data)), 
            replace=False
        )
        
        return disease_data.iloc[sample_indices]["symptoms"].tolist()
    
    def get_disease_statistics(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get statistics about diseases in the dataset.
        
        Returns:
            Dictionary with disease statistics.
        """
        if self.data is None:
            self.load_data()
        
        stats = {}
        for disease in self.disease_names:
            disease_data = self.data[self.data["disease"] == disease]
            
            # Count symptoms per case
            symptom_counts = disease_data["symptoms"].apply(len)
            
            stats[disease] = {
                "count": len(disease_data),
                "percentage": len(disease_data) / len(self.data) * 100,
                "avg_symptoms": symptom_counts.mean(),
                "min_symptoms": symptom_counts.min(),
                "max_symptoms": symptom_counts.max()
            }
        
        return stats
    
    def get_symptom_frequencies(self) -> Dict[str, int]:
        """Get frequency of each symptom across all cases.
        
        Returns:
            Dictionary mapping symptoms to their frequencies.
        """
        if self.data is None:
            self.load_data()
        
        symptom_counts = {}
        for symptoms in self.data["symptoms"]:
            for symptom in symptoms:
                symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1
        
        return dict(sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True))
