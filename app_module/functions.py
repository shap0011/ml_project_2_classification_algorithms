import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'info'

import logging

# Configure logging manually again
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')

# Load dataset
def load_data(filepath):
    logging.info(f"Loading data from {filepath}")
    # load the dataset from a CSV file
    df = pd.read_csv(filepath)
    logging.info(f"Data loaded successfully with shape {df.shape}")
    return df

def fill_missing_values(df):
    logging.info("Filling missing values...")
    # impute missing values in categorical variables with the most frequent value (mode) or a constant
    # fill missing 'Gender' values with 'Male'
    df['Gender'].fillna('Male', inplace=True)
    # fill missing 'Married' values with the most frequent value in the 'Married' column
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    # fill missing 'Dependents' values with the most frequent value in the 'Dependents' column
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    # fill missing 'Self_Employed' values with the most frequent value in the 'Self_Employed' column
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    # fill missing 'Loan_Amount_Term' values with the most frequent value in the 'Loan_Amount_Term' column
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    # fill missing 'Credit_History' values with the most frequent value in the 'Credit_History' column
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    # impute missing values in a numerical variable with the median
    # fill missing 'LoanAmount' values with the median loan amount
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    logging.info("Missing values filled successfully.")
    return df

# Prepare data
def prepare_data(df):
    # drop 'Loan_ID' variable from the data. We won't need it.
    df = df.drop('Loan_ID', axis=1)
    # Create dummy variables for all 'object' type variables except 'Loan_Status'
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], dtype=int)
    # replace values in Loan_approved column
    df['Loan_Approved'] = df['Loan_Approved'].replace({'Y': 1, 'N': 0})
    return df

def save_processed_data(df, path):
    logging.info(f"Saving processed dataset to {path}")
    # saving this processed dataset
    df.to_csv(path, index=False)
    logging.info(f"Dataset saved successfully to {path}")
    # st.success(f"Processed dataset saved to `{path}`.")

# Split into train/test
def split_data(df):
    # Separate the input features and target variable
    x = df.drop('Loan_Approved', axis=1)
    y = df['Loan_Approved']
    # splitting the data in training and testing set
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, stratify=y)
    return xtrain, xtest, ytrain, ytest

# Scale features
def scale_features(xtrain, xtest):
    scaler = MinMaxScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)
    return xtrain_scaled, xtest_scaled, scaler


# Train Logistic Regression
def train_logistic_regression(xtrain_scaled, ytrain):
    logging.info("Training Logistic Regression model...")
    # train a Logistic Regression model on the scaled training data
    model = LogisticRegression()
    model.fit(xtrain_scaled, ytrain)
    logging.info("Logistic Regression model trained successfully.")
    return model

# Train Random Forest
def train_random_forest(xtrain, ytrain):
    logging.info("Training Random Forest model...")
    # create a Random Forest model with specific hyperparameters:
    model = RandomForestClassifier(n_estimators=2, max_depth=2, max_features=10)
    # train (fit) the Random Forest model on the training data
    model.fit(xtrain, ytrain)
    logging.info("Random Forest model trained successfully.")
    return model

# Cross Validation  
def cross_validate_model(model, xtrain, ytrain, model_name):
    logging.info(f"Starting cross-validation for {model_name}")
    kfold = KFold(n_splits=5)
    # use cross-validation to evaluate the model
    scores = cross_val_score(model, xtrain, ytrain, cv=kfold)
    logging.info(f"Finished cross-validation for {model_name} - Mean Accuracy: {scores.mean():.4f}")

