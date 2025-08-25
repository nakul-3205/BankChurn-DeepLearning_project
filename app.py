import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn & Salary Prediction", layout="wide")

# ----------------- Classification -----------------
# Load the trained classification model
model_class = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('label_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('One_Hot_encoder.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler_class = pickle.load(file)

# ----------------- Regression -----------------
# Load the trained regression model
model_reg = tf.keras.models.load_model('regressionmodel.h5')

# Load encoders and scaler for regression
with open('label_gender_regression.pkl', 'rb') as file:
    label_encoder_gender_reg = pickle.load(file)

with open('One_Hot_encoder_regression.pkl', 'rb') as file:
    onehot_encoder_geo_reg = pickle.load(file)

with open('scaler_regression.pkl', 'rb') as file:
    scaler_reg = pickle.load(file)

# ----------------- Tabs -----------------
tab1, tab2 = st.tabs(["💼 Customer Churn", "💵 Salary Prediction"])

# ----------------- Tab 1: Classification -----------------
with tab1:
    st.header("Customer Churn Prediction")

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92)
        balance = st.number_input('💰 Balance')
        credit_score = st.number_input('🏦 Credit Score')

    with col2:
        estimated_salary = st.number_input('💵 Estimated Salary')
        tenure = st.slider('📅 Tenure', 0, 10)
        num_of_products = st.slider('🔢 Number of Products', 1, 4)
        has_cr_card = st.selectbox('💳 Has Credit Card', ['No', 'Yes'])
        is_active_member = st.selectbox('✅ Is Active Member', ['No', 'Yes'])

    # Map 'No'/'Yes' to 0/1 for model
    has_cr_card_val = 0 if has_cr_card == 'No' else 1
    is_active_member_val = 0 if is_active_member == 'No' else 1

    if st.button('Predict Churn'):
        # Prepare input
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card_val],
            'IsActiveMember': [is_active_member_val],
            'EstimatedSalary': [estimated_salary]
        })

        # Encode geography
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Scale
        input_data_scaled = scaler_class.transform(input_data)

        # Predict
        prediction = model_class.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        st.subheader("🔹 Prediction Results")
        st.write(f'Churn Probability: {prediction_proba:.2f}')

        churn_label = 'Yes' if prediction_proba > 0.5 else 'No'
        churn_val = 1 if prediction_proba > 0.5 else 0

        if churn_label == 'Yes':
            st.write('The customer is likely to churn')
        else:
            st.write('The customer is not likely to churn')

# ----------------- Tab 2: Regression -----------------
with tab2:
    st.header("Salary Prediction (Regression)")

    col1, col2 = st.columns(2)

    with col1:
        country_r = st.selectbox('🌍 Country', onehot_encoder_geo_reg.categories_[0], key="country_r")
        gender_r = st.selectbox('👤 Gender', label_encoder_gender_reg.classes_, key="gender_r")
        age_r = st.slider('🎂 Age', 18, 92, key="age_r")
        balance_r = st.number_input('💰 Balance', key="balance_r")
        credit_score_r = st.number_input('🏦 Credit Score', key="credit_r")

    with col2:
        tenure_r = st.slider('📅 Tenure', 0, 10, key="tenure_r")
        num_of_products_r = st.slider('🔢 Number of Products', 1, 4, key="numprod_r")
        has_cr_card_r = st.selectbox('💳 Has Credit Card', ['No', 'Yes'], key="crcard_r")
        is_active_member_r = st.selectbox('✅ Is Active Member', ['No', 'Yes'], key="active_r")
        exited_r = st.selectbox('🚪 Exited', ['No', 'Yes'], key="exited_r")

    # Map 'No'/'Yes' to 0/1 for model
    has_cr_card_val_r = 0 if has_cr_card_r == 'No' else 1
    is_active_member_val_r = 0 if is_active_member_r == 'No' else 1
    exited_val_r = 0 if exited_r == 'No' else 1

    if st.button('Predict Salary', key="predict_salary_btn"):
        # Prepare input
        input_data_r = pd.DataFrame({
            'CreditScore': [credit_score_r],
            'Gender': [label_encoder_gender_reg.transform([gender_r])[0]],
            'Age': [age_r],
            'Tenure': [tenure_r],
            'Balance': [balance_r],
            'NumOfProducts': [num_of_products_r],
            'HasCrCard': [has_cr_card_val_r],
            'IsActiveMember': [is_active_member_val_r],
            'Exited': [exited_val_r]
        })

        # Encode country as Geography for the model
        geo_encoded_r = onehot_encoder_geo_reg.transform([[country_r]]).toarray()
        geo_encoded_df_r = pd.DataFrame(geo_encoded_r, columns=onehot_encoder_geo_reg.get_feature_names_out(['Geography']))
        input_data_r = pd.concat([input_data_r.reset_index(drop=True), geo_encoded_df_r], axis=1)

        # Scale
        input_data_scaled_r = scaler_reg.transform(input_data_r)

        # Predict
        prediction_r = model_reg.predict(input_data_scaled_r)
        salary_pred = prediction_r[0][0]

        st.subheader("🔹 Prediction Results")
        st.write(f'Predicted Salary: {salary_pred:.2f}')
        st.write(f'Exited Status: {exited_r} ({exited_val_r})')
