# Loading all the necessary packages
import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'info'

import logging
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from app_module import functions as func

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

try: 
    
    #-------- page setting, header, intro ---------------------
       
    # Set page configuration
    st.set_page_config(page_title="🏦 Loan Eligibility Prediction App", layout="wide")
    
    # Define color variables
    header_color = "#c24d2c"

    # set the title of the Streamlit app    
    st.markdown(f"<h1 style='color: {header_color};'>🏦 Loan Eligibility Prediction App</h1>", unsafe_allow_html=True)
    
       
    #-------- the app overview -----------------------------
    
    st.markdown("""
    ### Overview
    Welcome to the Loan Eligibility Prediction App!

    This web application helps predict whether a loan applicant is likely to be approved 
    for a loan based on their personal and financial information.

    Using machine learning models trained on real-world data, the app analyzes inputs 
    like applicant income, loan amount, credit history, and more to determine the probability of loan approval.

    It provides instant feedback to assist banks, financial advisors, and applicants 
    in understanding eligibility chances before a formal application is submitted.
    """)
    
    #-------- user instructions -------------------------------
    
    st.markdown("""
    ### How to Use This App

    1. **Review the Overview**
       Get familiar with the purpose of the app and the key features considered in the prediction.

    2. **Input Applicant Details**

    - Fill in the required fields such as income, loan amount, credit history, property area, etc.

    - Answer simple yes/no questions (0 = No, 1 = Yes) for categorical features.

    3. **Submit the Form**

    - Click the "**Predict Loan Eligibility**" button after completing the form.

    4. **View Prediction Result**

    - The app will instantly display whether the loan is **approved** ✅ or **not approved*** ❌ based on your inputs.

    > 💡 *Tip: Try changing different fields (e.g., increase income or improve credit history) to see how it affects loan approval chances!*
    """)
    
    #-------- the dataset loading -----------------------------
    
    try:
        df = func.load_data('data/credit.csv')
        logging.info("Dataset loaded successfully!")
        logging.warning("[INFO] Dataset loaded successfully.") # for the Streamlit web app
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        st.error("Dataset file not found. Please check the 'data/final.csv' path.")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        st.error("An unexpected error occurred while loading the dataset.")
    
     #-------- loan approval distribution -----------------------------

    df['Credit_History'] = df['Credit_History'].astype('object')
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('object')

    # Fill missing values
    df = func.fill_missing_values(df)

    # create a copy of the original DataFrame and store it in 'raw'
    raw = df.copy()

    # Prepare data (drop ID, dummies, target encoding)
    df = func.prepare_data(df)

    # Save processed data
    func.save_processed_data(df, "data/Processed_Credit_Dataset.csv")

    # Split into train/test
    xtrain, xtest, ytrain, ytest = func.split_data(df)

    # Scale features
    xtrain_scaled, xtest_scaled, scaler = func.scale_features(xtrain, xtest)

    # Logistic Regression
    try:
        lrmodel = func.train_logistic_regression(xtrain_scaled, ytrain)
    except Exception as e:
        logging.error(f"Failed to train Logistic Regression: {e}", exc_info=True)
        st.error("Failed to train Logistic Regression model.")
    
    # ----------- USER INPUT FORM: Predict Loan Eligibility ------------

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🏦 Predict Loan Eligibility Based on Your Inputs")

    # Load the trained Logistic Regression model (use your lrmodel already trained above)
    model = lrmodel

    # Define the feature columns (important: these must match the training features)
    feature_names = xtrain.columns.tolist()

    with st.form(key="loan_prediction_form"):
        st.markdown("### Enter Applicant Details:")

        col1, col2 = st.columns(2)

        with col1:
            ApplicantIncome = st.number_input("Applicant Income:", min_value=0, value=5000)
            CoapplicantIncome = st.number_input("Coapplicant Income:", min_value=0, value=2000)
            LoanAmount = st.number_input("Loan Amount (in thousands):", min_value=0, value=150)
            Loan_Amount_Term_360 = st.selectbox("Loan Amount Term is 360 months?", options=[0, 1])
            Credit_History = st.selectbox("Credit History (1: Good, 0: Bad):", options=[1, 0])
            Property_Area_Semiurban = st.selectbox("Property Area is Semiurban?", options=[0, 1])

        with col2:
            Gender_Male = st.selectbox("Is Applicant Male?", options=[0, 1])
            Married_Yes = st.selectbox("Is Applicant Married?", options=[0, 1])
            Dependents_0 = st.selectbox("Dependents (0)?", options=[0, 1])
            Education_Not_Graduate = st.selectbox("Is Not Graduate?", options=[0, 1])
            Self_Employed_Yes = st.selectbox("Is Self Employed?", options=[0, 1])
            Property_Area_Urban = st.selectbox("Property Area is Urban?", options=[0, 1])

        submit_button = st.form_submit_button(label="Predict Loan Eligibility")

    # ----------- After Submit -------------
    if submit_button:
        try:
            # Create a DataFrame from user input
            user_input = pd.DataFrame([[ 
                ApplicantIncome,
                CoapplicantIncome,
                LoanAmount,
                Loan_Amount_Term_360,
                Credit_History,
                Property_Area_Semiurban,
                Gender_Male,
                Married_Yes,
                Dependents_0,
                Education_Not_Graduate,
                Self_Employed_Yes,
                Property_Area_Urban
            ]], columns=[
                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term_360',
                'Credit_History', 'Property_Area_Semiurban', 'Gender_Male', 'Married_Yes',
                'Dependents_0', 'Education_Not_Graduate', 'Self_Employed_Yes', 'Property_Area_Urban'
            ])

            # Add missing columns with 0
            for col in feature_names:
                if col not in user_input.columns:
                    user_input[col] = 0  # Add missing columns as 0

            # Reorder columns correctly
            user_input = user_input[feature_names]

            # Scale user input
            user_input_scaled = scaler.transform(user_input)

            # Make prediction
            prediction = model.predict(user_input_scaled)[0]

            if prediction == 1:
                st.success("✅ Congratulations! Loan Approved.")
            else:
                st.error("❌ Sorry, Loan Not Approved.")

        except Exception as e:
            logging.error(f"Prediction failed: {e}", exc_info=True)
            st.error("Prediction failed. Please check your inputs or try again.")

    
except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    st.error("Something went wrong! Please check the logs or try again later.")


