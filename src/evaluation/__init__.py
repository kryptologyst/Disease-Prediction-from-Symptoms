"""Evaluation metrics and analysis for healthcare AI models."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ClinicalMetrics:
    """Clinical evaluation metrics for disease prediction models."""
    
    def __init__(self, class_names: List[str]):
        """Initialize clinical metrics calculator.
        
        Args:
            class_names: List of disease class names.
        """
        self.class_names = class_names
        self.n_classes = len(class_names)
        
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities.
            
        Returns:
            Dictionary of metrics.
        """
        metrics = {}
        
        # Overall metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["macro_auc"] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        metrics["macro_auprc"] = average_precision_score(y_true, y_proba, average='macro')
        
        # Per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            # Binary classification for this class
            y_true_binary = (y_true == i).astype(int)
            y_proba_binary = y_proba[:, i]
            y_pred_binary = (y_pred == i).astype(int)
            
            # Calculate metrics
            if len(np.unique(y_true_binary)) > 1:  # Check if class exists in test set
                auc = roc_auc_score(y_true_binary, y_proba_binary)
                auprc = average_precision_score(y_true_binary, y_proba_binary)
            else:
                auc = 0.0
                auprc = 0.0
            
            # Confusion matrix for this class
            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
            
            # Clinical metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            
            class_metrics[class_name] = {
                "auc": auc,
                "auprc": auprc,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "ppv": ppv,
                "npv": npv,
                "support": np.sum(y_true == i)
            }
        
        metrics["per_class"] = class_metrics
        
        # Calibration metrics
        metrics["calibration_error"] = self._calculate_calibration_error(y_true, y_proba)
        
        return metrics
    
    def _calculate_calibration_error(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate expected calibration error.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            n_bins: Number of bins for calibration.
            
        Returns:
            Expected calibration error.
        """
        # Get predicted probabilities for true classes
        y_proba_true = y_proba[np.arange(len(y_true)), y_true]
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (y_proba_true > bin_lower) & (y_proba_true <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence in this bin
                accuracy_in_bin = y_proba_true[in_bin].mean()
                avg_confidence_in_bin = y_proba_true[in_bin].mean()
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            save_path: Path to save the plot.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot ROC curves for all classes.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            save_path: Path to save the plot.
        """
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            y_true_binary = (y_true == i).astype(int)
            y_proba_binary = y_proba[:, i]
            
            if len(np.unique(y_true_binary)) > 1:
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba_binary)
                auc = roc_auc_score(y_true_binary, y_proba_binary)
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot precision-recall curves for all classes.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            save_path: Path to save the plot.
        """
        plt.figure(figsize=(10, 8))
        
        # Plot PR curve for each class
        for i, class_name in enumerate(self.class_names):
            y_true_binary = (y_true == i).astype(int)
            y_proba_binary = y_proba[:, i]
            
            if len(np.unique(y_true_binary)) > 1:
                precision, recall, _ = precision_recall_curve(y_true_binary, y_proba_binary)
                auprc = average_precision_score(y_true_binary, y_proba_binary)
                plt.plot(recall, precision, label=f'{class_name} (AUPRC = {auprc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot calibration curve.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            save_path: Path to save the plot.
        """
        # Get predicted probabilities for true classes
        y_proba_true = y_proba[np.arange(len(y_true)), y_true]
        
        # Create binary labels for calibration curve (1 for correct prediction, 0 for incorrect)
        y_binary = np.ones(len(y_true))  # All predictions are "positive" (correct)
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_binary, y_proba_true, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve (Confidence vs Accuracy)')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, class_names: List[str]):
        """Initialize model evaluator.
        
        Args:
            class_names: List of disease class names.
        """
        self.class_names = class_names
        self.metrics_calculator = ClinicalMetrics(class_names)
        
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_plots: bool = True,
        output_dir: str = "assets"
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Evaluate model comprehensively.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test labels.
            save_plots: Whether to save plots.
            output_dir: Directory to save plots.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred, y_proba)
        
        # Print summary
        self._print_evaluation_summary(metrics)
        
        # Generate plots
        if save_plots:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            self.metrics_calculator.plot_confusion_matrix(
                y_test, y_pred, 
                save_path=os.path.join(output_dir, "confusion_matrix.png")
            )
            
            self.metrics_calculator.plot_roc_curves(
                y_test, y_proba,
                save_path=os.path.join(output_dir, "roc_curves.png")
            )
            
            self.metrics_calculator.plot_precision_recall_curves(
                y_test, y_proba,
                save_path=os.path.join(output_dir, "pr_curves.png")
            )
            
            self.metrics_calculator.plot_calibration_curve(
                y_test, y_proba,
                save_path=os.path.join(output_dir, "calibration_curve.png")
            )
        
        return metrics
    
    def _print_evaluation_summary(self, metrics: Dict[str, Union[float, Dict[str, float]]]) -> None:
        """Print evaluation summary.
        
        Args:
            metrics: Calculated metrics.
        """
        print("=" * 60)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro AUC: {metrics['macro_auc']:.4f}")
        print(f"Macro AUPRC: {metrics['macro_auprc']:.4f}")
        print(f"Calibration Error: {metrics['calibration_error']:.4f}")
        
        print("\nPer-Class Metrics:")
        print("-" * 60)
        print(f"{'Disease':<20} {'AUC':<8} {'AUPRC':<8} {'Sens':<8} {'Spec':<8} {'PPV':<8} {'NPV':<8}")
        print("-" * 60)
        
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"{class_name:<20} "
                  f"{class_metrics['auc']:<8.3f} "
                  f"{class_metrics['auprc']:<8.3f} "
                  f"{class_metrics['sensitivity']:<8.3f} "
                  f"{class_metrics['specificity']:<8.3f} "
                  f"{class_metrics['ppv']:<8.3f} "
                  f"{class_metrics['npv']:<8.3f}")
        
        print("=" * 60)
    
    def compare_models(
        self,
        models: Dict[str, any],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """Compare multiple models.
        
        Args:
            models: Dictionary of model names to model instances.
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            DataFrame with comparison results.
        """
        results = []
        
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred, y_proba)
            
            results.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Macro AUC': metrics['macro_auc'],
                'Macro AUPRC': metrics['macro_auprc'],
                'Calibration Error': metrics['calibration_error']
            })
        
        return pd.DataFrame(results).sort_values('Macro AUC', ascending=False)
