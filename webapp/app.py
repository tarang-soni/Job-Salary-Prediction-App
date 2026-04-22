import streamlit as st
import pandas as pd
import joblib
import os

# --- BULLETPROOF PATH SETUP ---
webapp_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(webapp_dir)

model_path = os.path.join(project_dir, 'data', 'salary_prediction_model.pkl')
csv_path = os.path.join(project_dir, 'dataset', 'job_salary_prediction_dataset.csv')

# --- CACHING TO IMPROVE PERFORMANCE ---

@st.cache_resource
def load_model(path):
    """Loads the model only once and keeps it in memory"""
    return joblib.load(path)

@st.cache_data
def load_data(path):
    """Loads the CSV only once and keeps it in memory"""
    return pd.read_csv(path)

# Load the files using the cached functions
model = load_model(model_path)
df = load_data(csv_path)

# --- UI AND APP LOGIC ---

st.title('Tech Job Salary Predictor 💼')
st.write("Enter the job details below to predict the estimated salary.")

st.markdown("---")

# Create Inputs side-by-side
st.subheader("Numerical Details")
col1, col2, col3 = st.columns(3)

with col1:
    experience_years = st.number_input('Years of Experience', min_value=0, max_value=50, value=5)
with col2:
    skills_count = st.number_input('Number of Skills', min_value=0, max_value=50, value=5)
with col3:
    certifications = st.number_input('Certifications', min_value=0, max_value=20, value=1)

st.subheader("Categorical Details")
col4, col5 = st.columns(2)

with col4:
    job_title = st.selectbox('Job Title', df['job_title'].unique())
    industry = st.selectbox('Industry', df['industry'].unique())
    location = st.selectbox('Location', df['location'].unique())

with col5:
    education_level = st.selectbox('Education Level', df['education_level'].unique())
    company_size = st.selectbox('Company Size', df['company_size'].unique())
    remote_work = st.selectbox('Remote Work', df['remote_work'].unique())

st.markdown("---")

if st.button('Predict Salary'):
    # Put the user's input into a pandas DataFrame
    input_data = pd.DataFrame({
        'job_title': [job_title],
        'experience_years': [experience_years],
        'education_level': [education_level],
        'skills_count': [skills_count],
        'industry': [industry],
        'company_size': [company_size],
        'location': [location],
        'remote_work': [remote_work],
        'certifications': [certifications]
    })
    
    # Recreate the 42 columns
    categorical_cols = ['job_title', 'education_level', 'industry', 'company_size', 'location', 'remote_work']
    
    X_original = df.drop(columns=['salary'])
    X_original_encoded = pd.get_dummies(X_original, columns=categorical_cols, drop_first=True)
    
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols)
    input_encoded = input_encoded.reindex(columns=X_original_encoded.columns, fill_value=0)
    
    # Make the Prediction
    prediction = model.predict(input_encoded)[0]
    
    # Display the result
    st.success(f"### Predicted Salary: ${prediction:,.2f}")