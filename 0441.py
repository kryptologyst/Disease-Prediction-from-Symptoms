#!/usr/bin/env python3
"""
Project 441: Disease Prediction from Symptoms - Modernized Healthcare AI Demo

This is a simple example demonstrating the modernized disease prediction system.
For the full implementation, see the src/ directory and run the training scripts.

⚠️ DISCLAIMER: This is a research demonstration only. NOT for clinical use.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import SymptomDataset
from src.models import SymptomPredictor
from src.utils import set_seed


def main():
    """Simple demonstration of the disease prediction system."""
    
    print("Disease Prediction from Symptoms - Demo")
    print("=" * 50)
    print("⚠️  RESEARCH DEMO ONLY - NOT FOR CLINICAL USE")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create dataset
    print("\n Creating synthetic dataset...")
    dataset = SymptomDataset(synthetic_size=500, random_state=42)
    dataset.load_data()
    
    print(f"Dataset created with {len(dataset.data)} samples")
    print(f"Found {len(dataset.symptom_names)} unique symptoms")
    print(f"Found {len(dataset.disease_names)} unique diseases")
    
    # Train model
    print("\n Training XGBoost model...")
    predictor = SymptomPredictor(model_type="xgboost", random_state=42)
    predictor.fit(dataset)
    
    # Evaluate model
    print("\n Evaluating model...")
    metrics = predictor.evaluate(dataset)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Average Precision: {metrics['average_precision']:.3f}")
    
    # Make predictions
    print("\n Making predictions...")
    
    test_cases = [
        ["fever", "cough", "sore throat"],
        ["headache", "nausea", "dizziness"],
        ["chest pain", "shortness of breath"],
        ["abdominal pain", "diarrhea", "vomiting"]
    ]
    
    for i, symptoms in enumerate(test_cases, 1):
        print(f"\nCase {i}: {', '.join(symptoms)}")
        
        # Predict disease
        predicted_disease = predictor.predict(symptoms)
        probabilities = predictor.predict_proba(symptoms)
        
        print(f"Predicted Disease: {predicted_disease}")
        print("All Probabilities:")
        
        # Show top 3 predictions
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for disease, prob in sorted_probs[:3]:
            print(f"  {disease}: {prob:.3f}")
    
    # Show feature importance
    print("\n Top 10 Most Important Features:")
    feature_importance = predictor.get_feature_importance()
    if feature_importance:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"{i:2d}. {feature:<20} {importance:.4f}")
    
    print("\n Demo completed!")
    print("\nFor the full interactive demo, run: streamlit run demo/app.py")
    print("For training scripts, see: scripts/train.py")
    print("For comprehensive evaluation, see: scripts/evaluate.py")


if __name__ == "__main__":
    main()
