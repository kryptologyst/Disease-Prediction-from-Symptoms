"""Explainability and interpretability for healthcare AI models."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Model explainability using SHAP and other methods."""
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        class_names: List[str],
        background_data: Optional[np.ndarray] = None
    ):
        """Initialize model explainer.
        
        Args:
            model: Trained model to explain.
            feature_names: Names of input features.
            class_names: Names of output classes.
            background_data: Background data for SHAP explainer.
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.background_data = background_data
        
        # Initialize SHAP explainer
        self.explainer = None
        self._setup_shap_explainer()
        
    def _setup_shap_explainer(self) -> None:
        """Setup SHAP explainer based on model type."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
            self.explainer = None
            return
            
        try:
            if hasattr(self.model, 'predict_proba'):
                # For sklearn models
                if self.background_data is not None:
                    self.explainer = shap.Explainer(
                        self.model.predict_proba,
                        self.background_data
                    )
                else:
                    # Use TreeExplainer for tree-based models
                    if hasattr(self.model, 'feature_importances_'):
                        self.explainer = shap.TreeExplainer(self.model)
                    else:
                        self.explainer = shap.Explainer(self.model.predict_proba)
            else:
                logger.warning("Model does not support SHAP explanation")
        except Exception as e:
            logger.warning(f"Failed to setup SHAP explainer: {e}")
            self.explainer = None
    
    def explain_prediction(
        self,
        symptoms: List[str],
        show_plot: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Explain a single prediction.
        
        Args:
            symptoms: List of input symptoms.
            show_plot: Whether to display plots.
            save_path: Path to save explanation plots.
            
        Returns:
            Dictionary with explanation results.
        """
        if self.explainer is None:
            logger.warning("SHAP explainer not available")
            return {}
        
        # Encode symptoms
        mlb = MultiLabelBinarizer()
        mlb.fit([self.feature_names])
        X = mlb.transform([symptoms])
        
        # Get SHAP values
        shap_values = self.explainer(X)
        
        # Get prediction
        prediction_proba = self.model.predict_proba(X)[0]
        predicted_class = np.argmax(prediction_proba)
        
        # Create explanation dictionary
        explanation = {
            'predicted_disease': self.class_names[predicted_class],
            'confidence': prediction_proba[predicted_class],
            'all_probabilities': dict(zip(self.class_names, prediction_proba)),
            'symptom_contributions': {}
        }
        
        # Extract symptom contributions
        if hasattr(shap_values, 'values'):
            # For newer SHAP versions
            values = shap_values.values[0]
        else:
            # For older SHAP versions
            values = shap_values[0]
        
        # Map SHAP values to symptom names
        for i, feature_name in enumerate(self.feature_names):
            if i < len(values):
                explanation['symptom_contributions'][feature_name] = float(values[i])
        
        # Sort by absolute contribution
        explanation['symptom_contributions'] = dict(
            sorted(
                explanation['symptom_contributions'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
        )
        
        # Generate plots
        if show_plot:
            self._plot_prediction_explanation(shap_values, symptoms, save_path)
        
        return explanation
    
    def _plot_prediction_explanation(
        self,
        shap_values,
        symptoms: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """Plot prediction explanation.
        
        Args:
            shap_values: SHAP values for the prediction.
            symptoms: Input symptoms.
            save_path: Path to save the plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Waterfall plot
        shap.waterfall_plot(shap_values[0], show=False)
        axes[0].set_title('SHAP Waterfall Plot')
        
        # Plot 2: Bar plot of feature contributions
        if hasattr(shap_values, 'values'):
            values = shap_values.values[0]
        else:
            values = shap_values[0]
        
        # Get top contributing features
        feature_contributions = list(zip(self.feature_names, values))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = feature_contributions[:10]  # Top 10 features
        
        feature_names_top = [f[0] for f in top_features]
        contributions = [f[1] for f in top_features]
        
        colors = ['red' if c < 0 else 'blue' for c in contributions]
        
        axes[1].barh(feature_names_top, contributions, color=colors)
        axes[1].set_xlabel('SHAP Value')
        axes[1].set_title('Top Feature Contributions')
        axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def explain_model_global(
        self,
        X: np.ndarray,
        show_plot: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, Union[float, List[float]]]:
        """Explain model globally using SHAP.
        
        Args:
            X: Input features.
            show_plot: Whether to display plots.
            save_path: Path to save explanation plots.
            
        Returns:
            Dictionary with global explanation results.
        """
        if self.explainer is None:
            logger.warning("SHAP explainer not available")
            return {}
        
        # Calculate SHAP values for sample of data
        sample_size = min(100, len(X))
        X_sample = X[np.random.choice(len(X), sample_size, replace=False)]
        
        shap_values = self.explainer(X_sample)
        
        # Calculate mean absolute SHAP values
        if hasattr(shap_values, 'values'):
            mean_shap_values = np.mean(np.abs(shap_values.values), axis=0)
        else:
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        
        # Create global explanation
        global_explanation = {
            'feature_importance': dict(zip(self.feature_names, mean_shap_values)),
            'top_features': []
        }
        
        # Sort features by importance
        sorted_features = sorted(
            global_explanation['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        global_explanation['top_features'] = [
            {'feature': name, 'importance': importance}
            for name, importance in sorted_features[:10]
        ]
        
        # Generate plots
        if show_plot:
            self._plot_global_explanation(shap_values, save_path)
        
        return global_explanation
    
    def _plot_global_explanation(
        self,
        shap_values,
        save_path: Optional[str] = None
    ) -> None:
        """Plot global model explanation.
        
        Args:
            shap_values: SHAP values for multiple samples.
            save_path: Path to save the plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Summary plot
        shap.summary_plot(shap_values, show=False)
        axes[0].set_title('SHAP Summary Plot')
        
        # Plot 2: Mean absolute SHAP values
        if hasattr(shap_values, 'values'):
            mean_shap_values = np.mean(np.abs(shap_values.values), axis=0)
        else:
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        
        # Get top features
        feature_importance = list(zip(self.feature_names, mean_shap_values))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_importance[:15]  # Top 15 features
        
        feature_names_top = [f[0] for f in top_features]
        importance_values = [f[1] for f in top_features]
        
        axes[1].barh(feature_names_top, importance_values, color='skyblue')
        axes[1].set_xlabel('Mean |SHAP Value|')
        axes[1].set_title('Global Feature Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_symptom_importance(
        self,
        dataset,
        top_n: int = 20
    ) -> Dict[str, Union[List[Dict], Dict[str, float]]]:
        """Analyze symptom importance across the dataset.
        
        Args:
            dataset: SymptomDataset instance.
            top_n: Number of top symptoms to analyze.
            
        Returns:
            Dictionary with symptom analysis results.
        """
        # Get symptom frequencies
        symptom_frequencies = dataset.get_symptom_frequencies()
        
        # Get global feature importance if available
        global_importance = {}
        if self.explainer is not None:
            X, _ = dataset.get_features()
            global_explanation = self.explain_model_global(X, show_plot=False)
            global_importance = global_explanation.get('feature_importance', {})
        
        # Combine frequency and importance
        analysis = {
            'symptom_frequency': symptom_frequencies,
            'symptom_importance': global_importance,
            'top_symptoms': []
        }
        
        # Create combined ranking
        combined_scores = {}
        for symptom in symptom_frequencies.keys():
            freq_score = symptom_frequencies[symptom] / max(symptom_frequencies.values())
            importance_score = global_importance.get(symptom, 0) / max(global_importance.values()) if global_importance else 0
            
            # Combined score (weighted average)
            combined_scores[symptom] = 0.6 * importance_score + 0.4 * freq_score
        
        # Sort by combined score
        sorted_symptoms = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        analysis['top_symptoms'] = [
            {
                'symptom': symptom,
                'frequency': symptom_frequencies[symptom],
                'importance': global_importance.get(symptom, 0),
                'combined_score': score
            }
            for symptom, score in sorted_symptoms[:top_n]
        ]
        
        return analysis
    
    def plot_symptom_analysis(
        self,
        analysis_results: Dict[str, Union[List[Dict], Dict[str, float]]],
        save_path: Optional[str] = None
    ) -> None:
        """Plot symptom analysis results.
        
        Args:
            analysis_results: Results from analyze_symptom_importance.
            save_path: Path to save the plot.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Symptom frequency
        top_symptoms = analysis_results['top_symptoms'][:10]
        symptom_names = [s['symptom'] for s in top_symptoms]
        frequencies = [s['frequency'] for s in top_symptoms]
        
        axes[0, 0].barh(symptom_names, frequencies, color='lightcoral')
        axes[0, 0].set_xlabel('Frequency')
        axes[0, 0].set_title('Top Symptoms by Frequency')
        
        # Plot 2: Symptom importance
        importances = [s['importance'] for s in top_symptoms]
        axes[0, 1].barh(symptom_names, importances, color='lightblue')
        axes[0, 1].set_xlabel('SHAP Importance')
        axes[0, 1].set_title('Top Symptoms by Importance')
        
        # Plot 3: Combined scores
        combined_scores = [s['combined_score'] for s in top_symptoms]
        axes[1, 0].barh(symptom_names, combined_scores, color='lightgreen')
        axes[1, 0].set_xlabel('Combined Score')
        axes[1, 0].set_title('Top Symptoms by Combined Score')
        
        # Plot 4: Scatter plot of frequency vs importance
        all_symptoms = analysis_results['symptom_frequency']
        all_importances = analysis_results['symptom_importance']
        
        frequencies_all = [all_symptoms.get(symptom, 0) for symptom in all_importances.keys()]
        importances_all = list(all_importances.values())
        
        axes[1, 1].scatter(frequencies_all, importances_all, alpha=0.6)
        axes[1, 1].set_xlabel('Frequency')
        axes[1, 1].set_ylabel('SHAP Importance')
        axes[1, 1].set_title('Frequency vs Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_explanation_report(
        self,
        symptoms: List[str],
        dataset,
        save_path: Optional[str] = None
    ) -> str:
        """Generate a comprehensive explanation report.
        
        Args:
            symptoms: Input symptoms.
            dataset: SymptomDataset instance.
            save_path: Path to save the report.
            
        Returns:
            Formatted explanation report.
        """
        # Get prediction explanation
        explanation = self.explain_prediction(symptoms, show_plot=False)
        
        # Get global analysis
        global_analysis = self.analyze_symptom_importance(dataset)
        
        # Generate report
        report = f"""
EXPLANATION REPORT
==================

Input Symptoms: {', '.join(symptoms)}

PREDICTION RESULTS:
- Predicted Disease: {explanation.get('predicted_disease', 'Unknown')}
- Confidence: {explanation.get('confidence', 0):.3f}

DISEASE PROBABILITIES:
"""
        
        for disease, prob in explanation.get('all_probabilities', {}).items():
            report += f"- {disease}: {prob:.3f}\n"
        
        report += f"""
SYMPTOM CONTRIBUTIONS:
"""
        
        for symptom, contribution in explanation.get('symptom_contributions', {}).items():
            direction = "supports" if contribution > 0 else "opposes"
            report += f"- {symptom}: {contribution:.3f} ({direction} prediction)\n"
        
        report += f"""
TOP SYMPTOMS BY IMPORTANCE:
"""
        
        for i, symptom_info in enumerate(global_analysis['top_symptoms'][:5], 1):
            report += f"{i}. {symptom_info['symptom']} "
            report += f"(Frequency: {symptom_info['frequency']}, "
            report += f"Importance: {symptom_info['importance']:.3f})\n"
        
        report += f"""
INTERPRETATION NOTES:
- Positive SHAP values indicate symptoms that support the predicted disease
- Negative SHAP values indicate symptoms that oppose the predicted disease
- Higher absolute values indicate stronger influence on the prediction
- This analysis is based on synthetic data and should not be used for clinical decisions
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
