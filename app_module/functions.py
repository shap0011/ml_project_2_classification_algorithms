import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'info'

import logging

# Configure logging manually again
logging.basicConfig(
    level=logging.DEBUG,
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

# Show project scope
def show_scope(header_color):
    # add the project scope and description
    st.markdown(f"<h2 style='color: {header_color};'>Project Scope</h2>", unsafe_allow_html=True)
    st.markdown("""
            Loans are a vital part of banking operations. However, not all loans are repaid, 
            making it crucial for banks to carefully monitor loan applications. 
            This case study analyzes the German Credit dataset, which contains information on 614 loan applicants, 
            including 13 attributes and the classification of whether each applicant was granted or denied a loan.

            **Your Role:**
            Using the available dataset, train a classification model to predict whether an applicant should be approved for a loan.

            **Goal:**
            Build a model that predicts loan eligibility with an average accuracy of over 76%.

            **Details:**
            - Machine Learning Task: Classification
            - Target Variable: `Loan_Approved`
            - Input Variables: Refer to the data dictionary below
            - Success Criteria: Achieve an accuracy of 76% or higher
    """)


# Show data dictionary
def show_data_dictionary(header_color):
    # add data dictionary
    st.markdown(f"<h2 style='color: {header_color};'>Data Dictionary:</h2>", unsafe_allow_html=True)
    st.markdown("""
            - **Loan_ID:** Applicant ID
            - **Gender:** Gender of the applicant Male/Female
            - **Married:** Marital status of the applicant
            - **Dependents:** Number of dependants the applicant has
            - **Education:** Highest level of education
            - **Self_Employed:** Whether self-employed Yes/No
            - **ApplicantIncome:** Income of the applicant per month
            - **CoapplicantIncome:** Income of the co-applicant per month
            - **LoanAmount:** Loan amount requested in *1000 dollars
            - **Loan_Amount_Term:** Term of the loan in months
            - **Credit_History:** Whether applicant has a credit history
            - **Property_Area:** Current property location
            - **Loan_Approved:** Loan approved yes/no
    """)


# Preview data
def preview_data(df, div_color):
    st.markdown(f"<h2 style='color: #c24d2c;'>Reading the data</h2>", unsafe_allow_html=True)
    st.write('First five and last five records:')
    # display the first five and last five rows of the dataset in the app
    st.write(df.head())
    st.write(df.tail())
    
    # create variables for rows and columns counts
    rows_count, columns_count = df.shape
    # display dataset shape
    st.markdown(f"""
    <div style='background-color: {div_color}; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
        The dataset contains:
        <ul>
            <li><strong>Rows:</strong> {rows_count}</li>
            <li><strong>Columns:</strong> {columns_count}</li>
        </ul>
    </div>
    <hr>
    """, unsafe_allow_html=True)

# Plot loan approval distribution
def plot_loan_approval_distribution(df):
    st.markdown("##### Loan Approval Distribution")
    # create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 3))
    # plot the bar chart
    df['Loan_Approved'].value_counts().plot.bar(ax=ax)
    # set labels and title
    ax.set_xlabel('Loan Approved')
    ax.set_ylabel('Count')
    ax.set_title('Loan Approval Distribution')
    # display the plot
    st.pyplot(fig)

# Display approval percentage
def display_approval_percentage(df, div_color):
    # Count Approved and Disapproved loans
    yes_count = df['Loan_Approved'].value_counts().get('Y', 0)
    no_count = df['Loan_Approved'].value_counts().get('N', 0)
    total = yes_count + no_count
    # round the result to one decimal place
    percent_approved = round(yes_count / total * 100, 1)
    
    # wrap it all in one div
    st.markdown(f"""
    <div style='background-color: {div_color}; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
        {yes_count} people ({percent_approved}%) out of {total} were approved for loan.
    </div>
    """, unsafe_allow_html=True)

# Show missing values
def show_missing_values(df, div_color):
    # check for missing values in each variable
    missing_values = df.isnull().sum()
    # loop through the missing values dictionary and print column names and their missing values side by side
    # build HTML string
    all_missing = ""
    for column, missing_count in missing_values.items():
        all_missing += f"<strong>{column}:</strong> {missing_count} missing<br>"
    # wrap it all in one div
    st.markdown(f"""
    <div style='background-color: {div_color}; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
        {all_missing}
    </div>
    """, unsafe_allow_html=True)

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
    st.success(f"Processed dataset saved to `{path}`.")

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
     # initialize the scaler
    scaler = MinMaxScaler()
    # fit the scaler on the training data and transform it
    xtrain_scaled = scaler.fit_transform(xtrain)
    # transform the testing data using the same scaler 
    xtest_scaled = scaler.transform(xtest)
    return xtrain_scaled, xtest_scaled

# Train Logistic Regression
def train_logistic_regression(xtrain_scaled, ytrain):
    logging.info("Training Logistic Regression model...")
    # train a Logistic Regression model on the scaled training data
    model = LogisticRegression()
    model.fit(xtrain_scaled, ytrain)
    logging.info("Logistic Regression model trained successfully.")
    return model


# Evaluate Logistic Regression
def evaluate_logistic_regression(model, xtest_scaled, ytest):
    logging.info("Evaluating Logistic Regression model...")
    # predict the loan eligibility on the scaled testing data
    ypred = model.predict(xtest_scaled)
    # calculate the accuracy score by comparing the predictions to the true labels
    acc = accuracy_score(ytest, ypred)
    logging.info(f"Logistic Regression Accuracy: {acc:.4f}")
    st.subheader("Logistic Regression Accuracy")
    # display the accuracy score in the Streamlit app
    st.write(f"Accuracy: `{acc:.4f}`")
    # create a DataFrame to compare actual vs predicted values
    st.dataframe(pd.DataFrame({'Actual': ytest, 'Predicted': ypred}))
    
    # display the confusion matrix
    cm = confusion_matrix(ytest, ypred)
    st.subheader("Confusion Matrix")
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))

# Train Random Forest
def train_random_forest(xtrain, ytrain):
    logging.info("Training Random Forest model...")
    # create a Random Forest model with specific hyperparameters:
    # n_estimators=2 -> number of trees in the forest
    # max_depth=2 -> maximum depth of each tree
    # max_features=10 -> number of features to consider when looking for the best split
    model = RandomForestClassifier(n_estimators=2, max_depth=2, max_features=10)
    # train (fit) the Random Forest model on the training data
    model.fit(xtrain, ytrain)
    logging.info("Random Forest model trained successfully.")
    return model

# Evaluate Random Forest
def evaluate_random_forest(model, xtest, ytest):
    logging.info("Evaluating Random Forest model...")
    # predict the target values (loan approval) on the testing data
    ypred = model.predict(xtest)
    # compute an accuracy score (how many correct predictions)
    acc = accuracy_score(ytest, ypred)
    logging.info(f"Random Forest Accuracy: {acc:.4f}")
    st.subheader("Random Forest Accuracy")
    # display the accuracy score
    st.write(f"Accuracy: `{acc:.4f}`")
    # compute the confusion matrix
    cm = confusion_matrix(ytest, ypred)
    st.subheader("Confusion Matrix")
    # turn it into a labeled DataFrame
    # display the confusion matrix
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))

# Cross Validation  
def cross_validate_model(model, xtrain, ytrain, model_name):
    logging.info(f"Starting cross-validation for {model_name}")
    st.subheader(f"Cross-Validation for {model_name}")
    # set up a KFold cross-validation
    kfold = KFold(n_splits=5)
    # use cross-validation to evaluate the model
    scores = cross_val_score(model, xtrain, ytrain, cv=kfold)
    # display the accuracy scores for each fold as a dataframe
    st.markdown("##### Accuracy Scores for Each Fold:")
    st.dataframe(pd.DataFrame(scores, columns=["Accuracy Score"]))
    # display the mean accuracy and standard deviation
    st.write(f"Scores: `{scores}`")
    st.write(f"Mean Accuracy: `{scores.mean():.4f}`")
    st.write(f"Standard Deviation: `{scores.std():.4f}`")
    logging.info(f"Finished cross-validation for {model_name} - Mean Accuracy: {scores.mean():.4f}")

