#!/usr/bin/env python3
"""Evaluation script for trained models."""

import argparse
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import SymptomDataset
from src.evaluation import ModelEvaluator
from src.explainability import ModelExplainer
from src.utils import setup_logging, ensure_dir


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--compare-models", action="store_true",
                       help="Compare multiple models if available")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Setup
    setup_logging(config.logging.level)
    ensure_dir(args.output_dir)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation process")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = SymptomDataset(
        synthetic_size=config.data.synthetic_size,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        include_metadata=config.data.include_metadata
    )
    
    # Split data
    X_train, X_test, y_train, y_test = dataset.split_data()
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    predictor = joblib.load(args.model_path)
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator(dataset.disease_names)
    metrics = evaluator.evaluate_model(
        predictor.model,
        X_test,
        y_test,
        save_plots=True,
        output_dir=args.output_dir
    )
    
    # Generate detailed analysis
    logger.info("Generating detailed analysis...")
    
    # Feature importance
    feature_importance = predictor.get_feature_importance()
    if feature_importance:
        importance_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        importance_path = os.path.join(args.output_dir, "feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")
    
    # Generate explanations
    if config.explainability.use_shap:
        logger.info("Generating explanations...")
        explainer = ModelExplainer(
            predictor.model,
            dataset.symptom_names,
            dataset.disease_names,
            background_data=X_train[:config.explainability.background_samples]
        )
        
        # Test on sample cases
        sample_cases = [
            ["fever", "cough", "sore throat"],
            ["headache", "nausea", "dizziness"],
            ["chest pain", "shortness of breath"],
            ["abdominal pain", "diarrhea", "vomiting"]
        ]
        
        explanations = []
        for i, symptoms in enumerate(sample_cases):
            logger.info(f"Explaining case {i+1}: {symptoms}")
            
            explanation = explainer.explain_prediction(
                symptoms,
                show_plot=False,
                save_path=os.path.join(args.output_dir, f"explanation_case_{i+1}.png")
            )
            
            explanations.append({
                'case_id': i+1,
                'symptoms': symptoms,
                'explanation': explanation
            })
        
        # Save explanations
        explanations_path = os.path.join(args.output_dir, "explanations.yaml")
        OmegaConf.save(explanations, explanations_path)
        logger.info(f"Explanations saved to {explanations_path}")
    
    # Generate summary report
    logger.info("Generating summary report...")
    
    report = f"""
EVALUATION SUMMARY REPORT
=========================

Model: {predictor.model_type}
Dataset Size: {config.data.synthetic_size}
Test Size: {len(X_test)}

OVERALL PERFORMANCE:
- Accuracy: {metrics['accuracy']:.4f}
- Macro AUC: {metrics['macro_auc']:.4f}
- Macro AUPRC: {metrics['macro_auprc']:.4f}
- Calibration Error: {metrics['calibration_error']:.4f}

PER-CLASS PERFORMANCE:
"""
    
    for disease, class_metrics in metrics['per_class'].items():
        report += f"""
{disease}:
  - AUC: {class_metrics['auc']:.3f}
  - AUPRC: {class_metrics['auprc']:.3f}
  - Sensitivity: {class_metrics['sensitivity']:.3f}
  - Specificity: {class_metrics['specificity']:.3f}
  - PPV: {class_metrics['ppv']:.3f}
  - NPV: {class_metrics['npv']:.3f}
  - Support: {class_metrics['support']}
"""
    
    report += f"""
TOP FEATURES BY IMPORTANCE:
"""
    
    if feature_importance:
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
            report += f"{i}. {feature}: {importance:.4f}\n"
    
    report += f"""
NOTES:
- This evaluation is based on synthetic data
- Results should not be interpreted as clinical performance
- Model is for research and educational purposes only
"""
    
    # Save report
    report_path = os.path.join(args.output_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Evaluation report saved to {report_path}")
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
