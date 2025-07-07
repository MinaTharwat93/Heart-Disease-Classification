# Heart Disease Classification Project

This project focuses on classifying the presence of heart disease in patients using various machine learning models. It includes data preprocessing, model training, hyperparameter tuning, and a Streamlit web application for prediction and visualization.

## Dataset

The dataset used is the **Heart Disease Dataset** from the UCI Machine Learning Repository. It contains 76 attributes, but this project utilizes a subset of 14 features commonly used in heart disease classification tasks.

**Key Features:**

*   **age**: The age of the patient in years.
*   **sex**: The gender of the patient (1 = male, 0 = female).
*   **cp**: The type of chest pain (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic).
*   **trestbps**: Resting blood pressure (in mm Hg) measured upon admission.
*   **chol**: Serum cholesterol level (in mg/dL).
*   **fbs**: Fasting blood sugar level (1 = >120 mg/dL, 0 = ≤120 mg/dL).
*   **restecg**: Resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = probable left ventricular hypertrophy).
*   **thalach**: Maximum heart rate achieved during a stress test.
*   **exang**: Exercise-induced angina (1 = yes, 0 = no).
*   **oldpeak**: ST depression induced by exercise relative to rest (in mm).
*   **slope**: The slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping).
*   **ca**: Number of major vessels (0-3) showing blockage on fluoroscopy.
*   **thal**: Thalassemia test result (3 = normal, 6 = fixed defect, 7 = reversible defect).
*   **num**: Presence or absence of heart disease (0 = no disease, 1-4 = varying degrees of disease). This is the target variable, which is binarized for this project (0 = no disease, 1 = presence of disease).

## Data Preprocessing

The following preprocessing steps were applied to the data:

*   **Handling Missing Values**:
    *   `SimpleImputer` with a median strategy was used for some numerical features.
    *   `KNNImputer` (with n_neighbors=5) was used for categorical features.
*   **Outlier Handling**:
    *   **Log Transformation**: Applied to features like 'trestbps', 'chol', and 'oldpeak' to handle skewness and outliers.
    *   **Lower/Upper Bound Capping**: Outliers for 'age' and 'thalach' were capped at 1.5 times the Interquartile Range (IQR) below the first quartile (Q1) or above the third quartile (Q3).
*   **Feature Scaling**:
    *   `RobustScaler` was used to scale numerical features, which is less sensitive to outliers.
*   **Dimensionality Reduction**:
    *   `Principal Component Analysis (PCA)` with `n_components=2` was applied to the preprocessed numerical features to reduce dimensionality while retaining significant variance.

## Machine Learning Models

Several classification models were implemented and evaluated:

*   Logistic Regression
*   Random Forest Classifier
*   XGBoost Classifier
*   Decision Tree Classifier
*   Support Vector Machine (SVM) Classifier

**Hyperparameter Tuning**:
`GridSearchCV` was utilized for tuning the hyperparameters of the models to find the optimal settings.

**Final Model**:
The Support Vector Machine (SVM) model with a linear kernel (C=2.0) was selected as the best-performing model based on evaluation metrics and was saved as `svm_pipeline_model.pkl` in the `models/` directory.

## How to Run the Project

**1. Dependencies:**

Ensure you have Python installed. The main dependencies are listed in `requirements.txt`:

*   pandas
*   numpy
*   matplotlib
*   seaborn
*   scikit-learn
*   scipy
*   streamlit
*   plotly

**2. Installation:**

Clone the repository and install the required packages:

```bash
git clone https://github.com/MinaTharwat93/Heart-Disease-Classification.git
cd Heart-Disease-Classification
pip install -r requirements.txt
```

**3. Running the Streamlit Application:**

To start the web application for predictions and visualizations, run:

```bash
streamlit run ui/app.py
```

This will open the application in your web browser.

## Project Structure

*   **`data/`**: Contains the raw dataset (`heart_disease.csv`).
*   **`models/`**: Stores the trained and saved machine learning model (`svm_pipeline_model.pkl`).
*   **`notebooks/`**: Includes Jupyter notebooks used for data exploration, preprocessing, model training, and evaluation (`Heart_Disease.ipynb`, `unsupervised_learning.ipynb`).
*   **`results/`**: Contains evaluation metrics and potentially other output files (`evaluation_metrics.txt`).
*   **`ui/`**: Holds the Streamlit application code (`app.py`).

## Results

The performance of the models was evaluated using metrics such as accuracy, log loss, classification reports (precision, recall, F1-score), and ROC AUC scores.

The final SVM model (referred to as Logistic Regression in the `evaluation_metrics.txt` due to being the last evaluated pipeline in the notebook before saving) achieved the following on the test set:

*   **Test Accuracy**: 0.8852
*   **Test Log Loss**: 0.3334

**Test Classification Report (for SVM):**

| Class | Precision | Recall | F1-Score | Support |
| :---- | :-------- | :----- | :------- | :------ |
| 0     | 0.87      | 0.90   | 0.88     | 29      |
| 1     | 0.90      | 0.88   | 0.89     | 32      |
| **Accuracy** |           |        | **0.89** | **61**  |
| **Macro Avg** | **0.88**  | **0.89**| **0.89** | **61**  |
| **Weighted Avg** | **0.89** | **0.89**| **0.89** | **61**  |

Detailed results and comparisons for other models can be found in the `Heart_Disease.ipynb` notebook and `results/evaluation_metrics.txt`.

## Visualizations (Streamlit App)

The Streamlit application (`ui/app.py`) provides interactive visualizations to explore the relationship between various features and the presence of heart disease. Available plot types include:

*   **Scatter Plot**: Visualize the distribution of a selected feature against heart disease status.
*   **Box Plot**: Compare the distribution of a selected feature for patients with and without heart disease.
*   **Histogram**: Show the frequency distribution of a selected feature.
*   **Bar Plot**: Display the average value of a selected feature grouped by heart disease status.

## About the Streamlit App

The "Heart Disease Prediction App" allows users to:

1.  **Explore Trends**: Visualize features from the original dataset to understand their relationship with heart disease.
2.  **Predict Heart Disease**: Input patient data through a user-friendly form. The app then uses the pre-trained SVM model to predict the probability of the patient having heart disease.

The app provides a simple interface for both data exploration and real-time prediction based on the developed machine learning pipeline.
