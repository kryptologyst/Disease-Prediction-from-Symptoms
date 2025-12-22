"""Streamlit demo application for disease prediction from symptoms."""

import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import SymptomDataset
from src.models import SymptomPredictor
from src.explainability import ModelExplainer
from src.utils import set_seed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Disease Prediction from Symptoms",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .symptom-tag {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        margin: 0.25rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None


def load_model():
    """Load the trained model and dataset."""
    try:
        # Check if model exists
        model_path = "models/model.pkl"
        if not os.path.exists(model_path):
            st.error("Model not found. Please train a model first using the training script.")
            return False
        
        # Load model
        predictor = joblib.load(model_path)
        
        # Create dataset
        dataset = SymptomDataset(synthetic_size=1000, random_state=42)
        dataset.load_data()
        
        # Create explainer
        X, _ = dataset.get_features()
        X_train, _, _, _ = dataset.split_data()
        
        explainer = ModelExplainer(
            predictor.model,
            dataset.symptom_names,
            dataset.disease_names,
            background_data=X_train[:100]
        )
        
        # Store in session state
        st.session_state.predictor = predictor
        st.session_state.dataset = dataset
        st.session_state.explainer = explainer
        st.session_state.model_loaded = True
        
        return True
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Disease Prediction from Symptoms</h1>', 
                unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ IMPORTANT DISCLAIMER:</strong><br>
        This is a research demonstration tool only. It is NOT intended for clinical use, 
        medical diagnosis, or treatment decisions. Always consult qualified healthcare 
        professionals for medical advice. This software uses synthetic data and should 
        not be used for real medical decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            if not load_model():
                st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Prediction", "Model Analysis", "Dataset Overview", "About"]
    )
    
    if page == "Prediction":
        prediction_page()
    elif page == "Model Analysis":
        analysis_page()
    elif page == "Dataset Overview":
        dataset_page()
    elif page == "About":
        about_page()


def prediction_page():
    """Prediction page."""
    st.header("🔮 Disease Prediction")
    
    # Get available symptoms
    dataset = st.session_state.dataset
    predictor = st.session_state.predictor
    explainer = st.session_state.explainer
    
    # Symptom selection
    st.subheader("Select Symptoms")
    
    # Multi-select for symptoms
    selected_symptoms = st.multiselect(
        "Choose symptoms:",
        options=dataset.symptom_names,
        default=["fever", "cough"],
        help="Select one or more symptoms to predict the disease"
    )
    
    # Prediction button
    if st.button("Predict Disease", type="primary"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            # Make prediction
            with st.spinner("Making prediction..."):
                try:
                    # Get prediction
                    predicted_disease = predictor.predict(selected_symptoms)
                    probabilities = predictor.predict_proba(selected_symptoms)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Prediction Results")
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Predicted Disease: {predicted_disease}</h4>
                            <p>Confidence: {probabilities[predicted_disease]:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### Selected Symptoms")
                        for symptom in selected_symptoms:
                            st.markdown(f'<span class="symptom-tag">{symptom}</span>', 
                                      unsafe_allow_html=True)
                    
                    # Probability distribution
                    st.markdown("### Disease Probabilities")
                    
                    # Sort probabilities
                    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                    
                    # Create bar chart
                    diseases = [item[0] for item in sorted_probs]
                    probs = [item[1] for item in sorted_probs]
                    
                    fig = px.bar(
                        x=probs,
                        y=diseases,
                        orientation='h',
                        title="Disease Probability Distribution",
                        labels={'x': 'Probability', 'y': 'Disease'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explanation
                    st.markdown("### Model Explanation")
                    
                    try:
                        explanation = explainer.explain_prediction(
                            selected_symptoms,
                            show_plot=False
                        )
                        
                        # Display symptom contributions
                        contributions = explanation.get('symptom_contributions', {})
                        
                        if contributions:
                            st.markdown("#### Symptom Contributions")
                            
                            # Create contribution chart
                            contrib_data = []
                            for symptom, contrib in contributions.items():
                                if symptom in selected_symptoms:
                                    contrib_data.append({
                                        'Symptom': symptom,
                                        'Contribution': contrib,
                                        'Type': 'Selected' if contrib > 0 else 'Selected (Negative)'
                                    })
                            
                            if contrib_data:
                                contrib_df = pd.DataFrame(contrib_data)
                                
                                fig_contrib = px.bar(
                                    contrib_df,
                                    x='Contribution',
                                    y='Symptom',
                                    color='Type',
                                    orientation='h',
                                    title="SHAP Values for Selected Symptoms",
                                    labels={'Contribution': 'SHAP Value', 'Symptom': 'Symptom'}
                                )
                                fig_contrib.add_vline(x=0, line_dash="dash", line_color="gray")
                                st.plotly_chart(fig_contrib, use_container_width=True)
                                
                                # Interpretation
                                st.markdown("#### Interpretation")
                                st.markdown("""
                                - **Positive SHAP values**: Symptoms that support the predicted disease
                                - **Negative SHAP values**: Symptoms that oppose the predicted disease
                                - **Higher absolute values**: Stronger influence on the prediction
                                """)
                        
                    except Exception as e:
                        st.warning(f"Could not generate explanation: {str(e)}")
                
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")


def analysis_page():
    """Model analysis page."""
    st.header("📊 Model Analysis")
    
    dataset = st.session_state.dataset
    predictor = st.session_state.predictor
    
    # Model information
    st.subheader("Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", predictor.model_type.title())
    
    with col2:
        st.metric("Number of Symptoms", len(dataset.symptom_names))
    
    with col3:
        st.metric("Number of Diseases", len(dataset.disease_names))
    
    # Feature importance
    st.subheader("Feature Importance")
    
    feature_importance = predictor.get_feature_importance()
    if feature_importance:
        # Create DataFrame
        importance_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        # Display top features
        top_n = st.slider("Number of top features to display:", 5, 20, 10)
        top_features = importance_df.head(top_n)
        
        # Create bar chart
        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Top {top_n} Most Important Features",
            labels={'Importance': 'Importance Score', 'Feature': 'Feature'}
        )
        fig.update_layout(height=max(300, top_n * 25))
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table
        st.markdown("#### Feature Importance Table")
        st.dataframe(top_features, use_container_width=True)
    
    # Dataset statistics
    st.subheader("Dataset Statistics")
    
    # Disease distribution
    disease_stats = dataset.get_disease_statistics()
    
    diseases = list(disease_stats.keys())
    counts = [stats['count'] for stats in disease_stats.values()]
    percentages = [stats['percentage'] for stats in disease_stats.values()]
    
    # Create pie chart
    fig_pie = px.pie(
        values=counts,
        names=diseases,
        title="Disease Distribution in Dataset"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Symptom frequency
    symptom_freq = dataset.get_symptom_frequencies()
    
    # Display top symptoms
    top_symptoms = list(symptom_freq.items())[:15]
    symptom_names = [item[0] for item in top_symptoms]
    frequencies = [item[1] for item in top_symptoms]
    
    fig_symptoms = px.bar(
        x=frequencies,
        y=symptom_names,
        orientation='h',
        title="Top 15 Most Frequent Symptoms",
        labels={'x': 'Frequency', 'y': 'Symptom'}
    )
    fig_symptoms.update_layout(height=500)
    st.plotly_chart(fig_symptoms, use_container_width=True)


def dataset_page():
    """Dataset overview page."""
    st.header("📋 Dataset Overview")
    
    dataset = st.session_state.dataset
    
    # Dataset information
    st.subheader("Dataset Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(dataset.data))
    
    with col2:
        st.metric("Unique Symptoms", len(dataset.symptom_names))
    
    with col3:
        st.metric("Unique Diseases", len(dataset.disease_names))
    
    with col4:
        avg_symptoms = dataset.data['symptoms'].apply(len).mean()
        st.metric("Avg Symptoms per Case", f"{avg_symptoms:.1f}")
    
    # Sample data
    st.subheader("Sample Data")
    
    # Display sample cases
    sample_size = st.slider("Number of samples to display:", 5, 50, 10)
    sample_data = dataset.data.sample(n=sample_size, random_state=42)
    
    # Format for display
    display_data = sample_data[['symptoms', 'disease']].copy()
    display_data['symptoms'] = display_data['symptoms'].apply(lambda x: ', '.join(x))
    display_data.columns = ['Symptoms', 'Disease']
    
    st.dataframe(display_data, use_container_width=True)
    
    # Disease details
    st.subheader("Disease Details")
    
    disease_stats = dataset.get_disease_statistics()
    
    # Create detailed table
    stats_data = []
    for disease, stats in disease_stats.items():
        stats_data.append({
            'Disease': disease,
            'Count': stats['count'],
            'Percentage': f"{stats['percentage']:.1f}%",
            'Avg Symptoms': f"{stats['avg_symptoms']:.1f}",
            'Min Symptoms': stats['min_symptoms'],
            'Max Symptoms': stats['max_symptoms']
        })
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)
    
    # Symptom details
    st.subheader("Symptom Details")
    
    symptom_freq = dataset.get_symptom_frequencies()
    
    # Create symptom table
    symptom_data = []
    for symptom, freq in symptom_freq.items():
        percentage = (freq / len(dataset.data)) * 100
        symptom_data.append({
            'Symptom': symptom,
            'Frequency': freq,
            'Percentage': f"{percentage:.1f}%"
        })
    
    symptom_df = pd.DataFrame(symptom_data)
    st.dataframe(symptom_df, use_container_width=True)


def about_page():
    """About page."""
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## Overview
    
    This application demonstrates a machine learning system for predicting diseases based on symptoms.
    It serves as an educational tool for understanding healthcare AI applications.
    
    ## Features
    
    - **Multiple ML Models**: Supports Logistic Regression, XGBoost, LightGBM, and Deep Neural Networks
    - **Interactive Prediction**: Real-time disease prediction from symptom input
    - **Model Explainability**: SHAP-based explanations for predictions
    - **Comprehensive Analysis**: Feature importance, dataset statistics, and model performance
    - **Visualization**: Interactive charts and plots for better understanding
    
    ## Technical Details
    
    - **Framework**: Streamlit for the web interface
    - **ML Libraries**: scikit-learn, XGBoost, LightGBM, PyTorch
    - **Visualization**: Plotly for interactive charts
    - **Explainability**: SHAP for model interpretation
    
    ## Dataset
    
    The application uses synthetic data generated with realistic symptom-disease mappings.
    The dataset includes:
    
    - 10 different diseases
    - 20+ unique symptoms
    - Patient metadata (age, gender, severity)
    - Balanced class distribution
    
    ## Model Performance
    
    Typical performance metrics on synthetic data:
    
    - **Accuracy**: 85-90%
    - **Macro AUC**: 0.85-0.90
    - **Macro AUPRC**: 0.80-0.85
    
    ## Limitations
    
    - Uses synthetic data only
    - Simplified symptom representation
    - Limited disease coverage
    - No temporal modeling
    - Not validated on real clinical data
    
    ## Disclaimer
    
    **This application is for research and educational purposes only.**
    
    - NOT intended for clinical use
    - NOT a medical device
    - NOT a substitute for professional medical advice
    - Results should not be used for medical decisions
    
    ## Contact
    
    For questions or issues, please refer to the project documentation or contact the development team.
    """)


if __name__ == "__main__":
    main()
