# Mental Health Happiness Prediction - Deep Learning Project 🧠💡

A comprehensive machine learning project that predicts happiness index based on mental health and lifestyle factors using multiple algorithms with fixed hyperparameters and regularization techniques.

## 📋 Project Overview

This project implements various machine learning algorithms to predict happiness scores (1-10 scale) based on mental health and lifestyle data. It includes:

- **Multiple ML Algorithms**: Linear Regression, Ridge, Lasso, Random Forest, XGBoost, LightGBM, SVR, Neural Networks
- **Fixed Hyperparameters**: Base values for consistent and reproducible results
- **Regularization**: Ridge, Lasso, Dropout, L1/L2 regularization
- **Comprehensive Evaluation**: RMSE, MAE, R², feature importance analysis
- **Deep Learning**: TensorFlow/Keras neural networks with dropout regularization

## 🗂️ Project Structure

```
MentalHealthLifestyleHabits/
│
├── Mental_Health_Lifestyle_Dataset.csv       # Dataset
├── data_exploration_and_ml_models.ipynb      # Main Jupyter notebook
├── requirements.txt                           # Dependencies
├── feature_names.txt                          # Generated feature names
├── .gitignore                                # Git ignore file
└── README.md                                 # Project documentation
```

## 📊 Dataset Features

The dataset contains the following features:

| Feature | Description | Type |
|---------|-------------|------|
| Country | Country of respondent | Categorical |
| Age | Age of respondent | Numerical |
| Gender | Gender identity | Categorical |
| Exercise Level | Physical activity level (Low/Moderate/High) | Ordinal |
| Diet Type | Type of diet (Balanced/Vegetarian/Vegan/Junk Food) | Categorical |
| Sleep Hours | Average sleep hours per day | Numerical |
| Stress Level | Self-reported stress level (Low/Moderate/High) | Ordinal |
| Mental Health Condition | Mental health status | Categorical |
| Work Hours per Week | Weekly working hours | Numerical |
| Screen Time per Day | Daily screen time in hours | Numerical |
| Social Interaction Score | Social interaction rating (1-10) | Numerical |
| **Happiness Score** | Target variable - Overall happiness (1-10) | **Numerical** |

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/limonyessi/MentalHealthLifestyleHabits.git
cd MentalHealthLifestyleHabits

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook

```bash
# Start Jupyter and open the main notebook
jupyter notebook data_exploration_and_ml_models.ipynb
```

**Note**: All machine learning models and analysis are contained within the Jupyter notebook. Simply run all cells sequentially to:
- Load and explore the dataset
- Preprocess the data with feature engineering
- Train multiple ML models with fixed hyperparameters
- Evaluate and compare model performance

## 🤖 Machine Learning Algorithms

### 1. Linear Models with Regularization

- **Linear Regression**: Baseline model
- **Ridge Regression**: L2 regularization (α = 1.0)
- **Lasso Regression**: L1 regularization (α = 0.1)

### 2. Tree-Based Models

- **Random Forest**: Ensemble method with fixed parameters
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 2

### 3. Gradient Boosting

- **XGBoost**: Extreme gradient boosting
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1
- **LightGBM**: Fast gradient boosting
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1

### 4. Support Vector Regression

- **SVR**: Non-linear regression
  - kernel: 'rbf'
  - C: 1.0
  - epsilon: 0.1

### 5. Neural Networks

- **Deep Neural Network** (TensorFlow/Keras):
  - Architecture: 128-64-32 neurons
  - Dropout regularization: 0.3
  - Early stopping
  - Adam optimizer (learning_rate: 0.001)

## 📈 Model Evaluation

The project includes comprehensive evaluation metrics:

- **Regression Metrics**: RMSE, MAE, R²
- **Model Comparison**: Performance ranking across all algorithms
- **Feature Importance**: Analysis from Random Forest, XGBoost, and LightGBM
- **Visualizations**: Training history, predicted vs actual plots, feature importance charts
- **Best Model Selection**: Automatic identification of top-performing model

## 🎯 Usage Examples

### Running the Analysis

1. **Open the Jupyter Notebook**:

   ```bash
   jupyter notebook data_exploration_and_ml_models.ipynb
   ```

2. **Execute All Cells**: Run cells sequentially from top to bottom, or use "Run All" from the Cell menu.

### Key Notebook Sections

- **Data Loading**: Import and explore the Mental Health Lifestyle Dataset
- **Data Preprocessing**: Feature engineering, encoding, and scaling
- **Model Training**: Train 7 different ML algorithms with fixed hyperparameters
- **Model Evaluation**: Compare performance using RMSE, MAE, and R² scores
- **Feature Analysis**: Examine feature importance from tree-based models
- **Visualization**: Charts showing model performance and data insights

### Making Predictions with Trained Models

After running the notebook, you can use any trained model for predictions:

```python
# Example using the best model (after running the notebook)
best_model_name = results_df.iloc[0]['Model']
best_model = models_results[best_model_name]['model']

# Prepare new data (same preprocessing as training)
# ... preprocessing steps ...

# Make prediction
prediction = best_model.predict(new_data_scaled)
print(f"Predicted Happiness Score: {prediction[0]:.2f}/10.0")
```

## 📊 Results and Performance

After running the notebook, you'll see:

- **Model Comparison Table**: RMSE, MAE, and R² scores for all models
- **Performance Visualizations**: Bar charts comparing model metrics
- **Feature Importance Plots**: Top features identified by tree-based models
- **Predicted vs Actual Plots**: Scatter plots showing model accuracy
- **Training History**: Neural network loss and MAE curves

### Expected Performance

- **Best RMSE**: Varies by model performance on your dataset
- **Best R²**: Model comparison will identify the top performer
- **Automatic Ranking**: Models are sorted by RMSE for easy comparison

## 🔧 Customization

### Adding New Features

1. Modify the feature engineering section in the notebook (Cell 8)
2. Add new derived features based on existing columns
3. Update preprocessing steps as needed
4. Re-run all subsequent cells

### Adding New Models

1. Add new model training code in a new cell
2. Follow the existing pattern for model evaluation
3. Update the `models_results` dictionary
4. Include the new model in comparison visualizations

### Adjusting Hyperparameters

Modify the fixed parameter values in the respective model training cells:

```python
# Example: Modify XGBoost parameters in Cell 15
xgb_model = xgb.XGBRegressor(
    n_estimators=200,  # Change from 100
    max_depth=8,       # Change from 6
    learning_rate=0.05, # Change from 0.1
    reg_alpha=0.1,     # Change from 0
    reg_lambda=1,
    random_state=42
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: Mental Health Lifestyle Dataset
- Libraries: scikit-learn, TensorFlow, XGBoost, LightGBM
- Visualization: matplotlib, seaborn, plotly
- Deep Learning: TensorFlow/Keras

## 📞 Contact

Your Name - [your.email@example.com]
Project Link: [https://github.com/limonyessi/MentalHealthLifestyleHabits](https://github.com/limonyessi/MentalHealthLifestyleHabits)

---

## Happy Learning! 🚀
