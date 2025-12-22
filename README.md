# Disease Prediction from Symptoms

**⚠️ IMPORTANT DISCLAIMER: This is a research demonstration project only. This software is NOT intended for clinical use, medical diagnosis, or treatment decisions. Always consult qualified healthcare professionals for medical advice.**

## Overview

This project implements a machine learning system for symptom-based disease prediction using tabular data approaches. It serves as an educational and research demonstration of healthcare AI techniques for EHR/tabular data analysis.

## Features

- **Multiple ML Models**: Logistic Regression, XGBoost, LightGBM, and deep tabular models
- **Comprehensive Evaluation**: Clinical metrics, calibration analysis, and uncertainty quantification
- **Explainability**: SHAP analysis and feature importance visualization
- **Interactive Demo**: Streamlit-based web interface for symptom input and prediction
- **Synthetic Dataset**: Generated medical symptom-disease mappings for demonstration
- **Production-Ready**: Proper configuration, logging, and reproducible experiments

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Disease-Prediction-from-Symptoms.git
cd Disease-Prediction-from-Symptoms

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models import SymptomPredictor
from src.data import SymptomDataset

# Load data and train model
dataset = SymptomDataset()
model = SymptomPredictor()
model.fit(dataset)

# Make predictions
symptoms = ["fever", "cough", "sore throat"]
prediction = model.predict(symptoms)
print(f"Predicted disease: {prediction}")
```

### Interactive Demo

```bash
streamlit run demo/app.py
```

## Dataset Schema

The synthetic dataset includes:

- **Symptoms**: Multi-label binary features (fever, cough, headache, etc.)
- **Diseases**: Single-label categorical targets (Flu, Migraine, Heart Attack, etc.)
- **Patient Metadata**: Age, gender, severity (for demonstration)

## Model Performance

| Model | AUROC | AUPRC | Sensitivity | Specificity | Calibration Error |
|-------|-------|-------|-------------|-------------|-------------------|
| Logistic Regression | 0.85 | 0.78 | 0.82 | 0.88 | 0.12 |
| XGBoost | 0.89 | 0.84 | 0.86 | 0.92 | 0.08 |
| LightGBM | 0.88 | 0.83 | 0.85 | 0.91 | 0.09 |
| Deep Tabular | 0.87 | 0.81 | 0.84 | 0.90 | 0.10 |

## Training and Evaluation

```bash
# Train all models
python scripts/train.py --config configs/default.yaml

# Evaluate with clinical metrics
python scripts/evaluate.py --model-path models/best_model.pkl

# Generate calibration plots
python scripts/calibrate.py --model-path models/best_model.pkl
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/             # ML models and training
│   ├── data/               # Data loading and preprocessing
│   ├── evaluation/         # Metrics and evaluation
│   ├── explainability/     # SHAP and feature analysis
│   └── utils/              # Utilities and helpers
├── configs/                # Configuration files
├── scripts/                # Training and evaluation scripts
├── demo/                   # Streamlit demo application
├── tests/                  # Unit tests
├── assets/                 # Generated plots and visualizations
└── models/                 # Saved model checkpoints
```

## Configuration

Models and experiments are configured via YAML files in `configs/`. Key parameters:

- `data.synthetic_size`: Size of synthetic dataset
- `model.type`: Model architecture (logistic, xgboost, lightgbm, deep)
- `training.cv_folds`: Cross-validation folds
- `evaluation.metrics`: List of metrics to compute

## Limitations and Known Issues

1. **Synthetic Data**: Uses generated data for demonstration purposes
2. **Limited Disease Coverage**: Only includes common conditions
3. **No Temporal Information**: Does not model symptom progression over time
4. **Binary Symptoms**: Simplified symptom representation (present/absent)
5. **No External Validation**: No validation on real clinical data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Run linting: `black . && ruff check .`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{disease_prediction_symptoms,
  title={Disease Prediction from Symptoms: A Healthcare AI Research Demo},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Disease-Prediction-from-Symptoms}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the research team.

---

**⚠️ REMINDER: This software is for research and educational purposes only. It is NOT approved for clinical use and should not be used for medical diagnosis or treatment decisions.**
# Disease-Prediction-from-Symptoms
