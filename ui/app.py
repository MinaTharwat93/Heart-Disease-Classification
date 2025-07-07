import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.express as px

# Define custom transformers
class LogTransformer:
    def fit(self, x, y=None):
        self.n_features_in_ = x.shape[1]
        return self
    def transform(self, x, y=None):
        assert self.n_features_in_ == x.shape[1]
        return np.log1p(x)

class Handle_outlier_lb_ub:
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        q1 = np.percentile(X, 25)
        q3 = np.percentile(X, 75)
        iqr = q3 - q1
        self.ub_train = q3 + 1.5 * iqr
        self.lb_train = q1 - 1.5 * iqr
        return self
    def transform(self, X, y=None):
        assert self.n_features_in_ == X.shape[1]
        X[X > self.ub_train] = self.ub_train
        X[X < self.lb_train] = self.lb_train
        return X

# Load the pre-trained model
try:
    with open('https://github.com/MinaTharwat93/Heart-Disease-Classification/blob/main/models/svm_pipeline_model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.sidebar.success("Model loaded successfully!")
except FileNotFoundError:
    st.sidebar.error("Model file 'https://github.com/MinaTharwat93/Heart-Disease-Classification/blob/main/models/svm_pipeline_model.pkl' not found. Please train and save the model first.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Error loading model: {str(e)}")
    st.stop()

# Load the original dataset
try:
    df = pd.read_csv(r'https://github.com/MinaTharwat93/Heart-Disease-Classification/blob/main/data/heart_disease.csv')
    # Verify that 'num' column exists and contains valid values
    if 'num' not in df.columns or df['num'].isnull().all():
        st.error("The dataset is missing the 'num' column or contains only NaN values. Please check the data.")
        st.stop()
    df = df.dropna(subset=['num'])  # Drop rows where 'num' is NaN
    # Convert 'num' to binary: 0 if 0, 1 if > 0
    df['num_binary'] = (df['num'] > 0).astype(int)
except FileNotFoundError:
    st.error("Dataset file 'https://github.com/MinaTharwat93/Heart-Disease-Classification/blob/main/data/heart_disease.csv' not found. Please check the path and ensure the file exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.stop()

# Data visualization section
st.header("Explore Heart Disease Trends")
feature_options = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
plot_types = ['Scatter Plot', 'Box Plot', 'Histogram', 'Bar Plot']
selected_feature = st.selectbox("Select Feature to Visualize", feature_options)
selected_plot = st.selectbox("Select Plot Type", plot_types)

# Create different visualizations based on selection using 'num_binary'
if selected_plot == 'Scatter Plot':
    fig = px.scatter(df, x=selected_feature, y='num_binary', color='num_binary',
                     color_continuous_scale=px.colors.sequential.Viridis,
                     labels={'num_binary': 'Heart Disease (0 = No, 1 = Yes)', selected_feature: selected_feature.capitalize()},
                     title=f'Heart Disease vs {selected_feature.capitalize()}')
    fig.update_layout(yaxis_title="Heart Disease (0 = No, 1 = Yes)")  # Ensure y-axis label is clear
elif selected_plot == 'Box Plot':
    fig = px.box(df, x='num_binary', y=selected_feature, color='num_binary',
                 labels={'num_binary': 'Heart Disease (0 = No, 1 = Yes)', selected_feature: selected_feature.capitalize()},
                 title=f'Box Plot: {selected_feature.capitalize()} by Heart Disease')
    fig.update_layout(xaxis_title="Heart Disease (0 = No, 1 = Yes)")  # Ensure x-axis label is clear
elif selected_plot == 'Histogram':
    fig = px.histogram(df, x=selected_feature, nbins=20,
                       labels={selected_feature: selected_feature.capitalize()},
                       title=f'Histogram of {selected_feature.capitalize()}')
elif selected_plot == 'Bar Plot':
    fig = px.bar(df, x='num_binary', y=selected_feature, color='num_binary',
                 labels={'num_binary': 'Heart Disease (0 = No, 1 = Yes)', selected_feature: 'Average ' + selected_feature.capitalize()},
                 title=f'Average {selected_feature.capitalize()} by Heart Disease',
                 barmode='group')
    fig.update_layout(xaxis_title="Heart Disease (0 = No, 1 = Yes)")  # Ensure x-axis label is clear

st.plotly_chart(fig)

# Streamlit app for prediction
st.title("Heart Disease Prediction App")

st.header("Enter Patient Data")
age = st.number_input("Age (years)", min_value=0, max_value=120, value=50)
trestbps = st.number_input("Resting Blood Pressure (trestbps, mm Hg)", min_value=0, value=120)
chol = st.number_input("Cholesterol (chol, mg/dL)", min_value=0, value=200)
thalach = st.number_input("Maximum Heart Rate (thalach)", min_value=0, value=150)

# Define mapping dictionaries for categorical features
sex_options = {"Female": 0, "Male": 1}
cp_options = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
fbs_options = {"≤120 mg/dL": 0, ">120 mg/dL": 1}
restecg_options = {"Normal": 0, "ST-T abnormality": 1, "Left ventricular hypertrophy": 2}
exang_options = {"No": 0, "Yes": 1}
slope_options = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
ca_options = {"0 vessels": 0, "1 vessel": 1, "2 vessels": 2, "3 vessels": 3}
thal_options = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}

# Input fields with descriptive labels
sex = st.selectbox("Sex", options=list(sex_options.keys()), format_func=lambda x: x)
cp = st.selectbox("Chest Pain Type", options=list(cp_options.keys()), format_func=lambda x: x)
fbs = st.selectbox("Fasting Blood Sugar", options=list(fbs_options.keys()), format_func=lambda x: x)
restecg = st.selectbox("Resting ECG", options=list(restecg_options.keys()), format_func=lambda x: x)
exang = st.selectbox("Exercise Induced Angina", options=list(exang_options.keys()), format_func=lambda x: x)
oldpeak = st.number_input("ST Depression (oldpeak, mm)", min_value=0.0, value=1.0)
slope = st.selectbox("Slope of ST Segment", options=list(slope_options.keys()), format_func=lambda x: x)
ca = st.selectbox("Number of Major Vessels", options=list(ca_options.keys()), format_func=lambda x: x)
thal = st.selectbox("Thalassemia", options=list(thal_options.keys()), format_func=lambda x: x)

# Convert selected options to numbers for the model
sex_num = sex_options[sex]
cp_num = cp_options[cp]
fbs_num = fbs_options[fbs]
restecg_num = restecg_options[restecg]
exang_num = exang_options[exang]
slope_num = slope_options[slope]
ca_num = ca_options[ca]
thal_num = thal_options[thal]

# Create input data as a pandas DataFrame with column names
input_data = pd.DataFrame(
    [[age, trestbps, chol, thalach, sex_num, cp_num, fbs_num, restecg_num, exang_num, oldpeak, slope_num, ca_num, thal_num]],
    columns=['age', 'trestbps', 'chol', 'thalach', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
)

# Predict button
if st.button("Predict Heart Disease"):
    try:
        # Preprocess and predict probability
        probability = model.predict_proba(input_data)[:, 1][0]
        
        # Display result
        st.subheader("Prediction Result")
        st.write(f"Probability of Heart Disease: {probability:.2f}")
        if probability > 0.5:
            st.write("**Prediction: Has Heart Disease**")
        else:
            st.write("**Prediction: No Heart Disease**")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add some info
st.sidebar.header("About")
st.sidebar.write("This app uses a pre-trained SVM model to predict the probability of heart disease and visualize trends.")
