#!/usr/bin/env python3
"""Training script for disease prediction models."""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import SymptomDataset
from src.models import SymptomPredictor
from src.evaluation import ModelEvaluator
from src.explainability import ModelExplainer
from src.utils import set_seed, get_device, setup_logging, ensure_dir


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train disease prediction models")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model-type", type=str, default=None,
                       help="Override model type from config")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    if args.model_type:
        config.model.type = args.model_type
    if args.seed:
        config.training.random_state = args.seed
        config.data.random_state = args.seed
    
    # Setup
    set_seed(config.training.random_state)
    setup_logging(config.logging.level)
    ensure_dir(args.output_dir)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting training process")
    logger.info(f"Configuration: {config}")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
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
    
    # Initialize model
    logger.info(f"Initializing {config.model.type} model...")
    predictor = SymptomPredictor(
        model_type=config.model.type,
        random_state=config.training.random_state,
        device=device
    )
    
    # Train model
    logger.info("Training model...")
    predictor.fit(dataset)
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator(dataset.disease_names)
    metrics = evaluator.evaluate_model(
        predictor.model,
        X_test,
        y_test,
        save_plots=config.evaluation.save_plots,
        output_dir=os.path.join(args.output_dir, "plots")
    )
    
    # Generate explanations
    if config.explainability.use_shap:
        logger.info("Generating model explanations...")
        explainer = ModelExplainer(
            predictor.model,
            dataset.symptom_names,
            dataset.disease_names,
            background_data=X_train[:config.explainability.background_samples]
        )
        
        # Global explanation
        global_explanation = explainer.explain_model_global(
            X_test[:50],  # Use subset for efficiency
            show_plot=True,
            save_path=os.path.join(args.output_dir, "global_explanation.png")
        )
        
        # Symptom analysis
        symptom_analysis = explainer.analyze_symptom_importance(dataset)
        explainer.plot_symptom_analysis(
            symptom_analysis,
            save_path=os.path.join(args.output_dir, "symptom_analysis.png")
        )
        
        # Generate explanation report
        sample_symptoms = ["fever", "cough", "sore throat"]
        explanation_report = explainer.generate_explanation_report(
            sample_symptoms,
            dataset,
            save_path=os.path.join(args.output_dir, "explanation_report.txt")
        )
        
        logger.info("Explanation report generated")
    
    # Save model
    if config.training.save_model:
        model_path = os.path.join(args.output_dir, "model.pkl")
        import joblib
        joblib.dump(predictor, model_path)
        logger.info(f"Model saved to {model_path}")
    
    # Save results (convert numpy types to Python types for OmegaConf)
    def convert_numpy_types(obj):
        """Convert numpy types to Python types for serialization."""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    metrics_serializable = convert_numpy_types(metrics)
    results_path = os.path.join(args.output_dir, "results.yaml")
    OmegaConf.save(metrics_serializable, results_path)
    logger.info(f"Results saved to {results_path}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
