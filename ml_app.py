import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- Page Config ---
st.set_page_config(page_title="Credit Risk ML Engine", layout="wide")

st.title("🤖 ML Credit Risk Scoring Engine")
st.markdown("""
This is a **Supervised Machine Learning** prototype to predict loan defaults. 
It utilizes a **Random Forest Classifier** to analyze applicant demographics and financial history, 
providing explainable risk scores.
""")

# --- Sidebar: Generate Synthetic Data ---
st.sidebar.header("Dataset Configuration")
n_samples = st.sidebar.slider("Sample Size", 1000, 10000, 5000)

@st.cache_data
def load_data(n):
    # Generate synthetic credit data
    np.random.seed(42)
    data = pd.DataFrame({
        'Income': np.random.normal(50000, 15000, n),
        'Age': np.random.randint(21, 70, n),
        'Loan_Amount': np.random.normal(15000, 5000, n),
        'Credit_History_Length': np.random.randint(1, 20, n),
        'Debt_to_Income': np.random.uniform(0.1, 0.9, n),
        'Previous_Defaults': np.random.choice([0, 1], n, p=[0.8, 0.2])
    })
    
    # Create a target variable (Default) based on logic + noise
    # High Debt, Low Income, Previous Defaults increase risk
    risk_score = (
        (data['Debt_to_Income'] * 10) +
        (data['Previous_Defaults'] * 5) -
        (data['Income'] / 10000) +
        (data['Loan_Amount'] / 5000)
    )
    # Add random noise
    risk_score += np.random.normal(0, 2, n)
    
    # Sigmoid to probability
    prob = 1 / (1 + np.exp(-risk_score))
    data['Default'] = (prob > 0.65).astype(int) # Imbalanced dataset
    
    return data

df = load_data(n_samples)

if st.checkbox("Show Raw Training Data"):
    st.dataframe(df.head())

# --- Model Training ---
X = df.drop('Default', axis=1)
y = df['Default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# --- Performance Dashboard ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Performance")
    acc = clf.score(X_test, y_test)
    st.metric("Model Accuracy", f"{acc:.2%}")
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Repaid', 'Default'], y=['Repaid', 'Default'],
                       title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

with col2:
    st.subheader("Feature Importance")
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': clf.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h',
                     title="Global Feature Importance (Gini Impurity)")
    st.plotly_chart(fig_imp, use_container_width=True)

# --- Live Prediction ---
st.markdown("---")
st.subheader("⚡ Live Risk Scoring")

c1, c2, c3 = st.columns(3)
p_income = c1.number_input("Applicant Income", 20000, 200000, 60000)
p_debt = c2.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.4)
p_loan = c3.number_input("Loan Amount", 1000, 50000, 15000)
p_age = c1.slider("Age", 21, 75, 30)
p_history = c2.slider("Credit History (Years)", 0, 30, 5)
p_defaults = c3.selectbox("Previous Defaults", [0, 1])

input_data = pd.DataFrame([[p_income, p_age, p_loan, p_history, p_debt, p_defaults]], columns=X.columns)

if st.button("Assess Risk"):
    prediction = clf.predict(input_data)[0]
    probability = clf.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"High Risk Detected! (Default Probability: {probability:.1%})")
    else:
        st.success(f"Low Risk - Loan Approved (Default Probability: {probability:.1%})")
