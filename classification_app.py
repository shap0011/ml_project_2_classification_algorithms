# Loading all the necessary packages
# import Streamlit for building web app
import streamlit as st
# import pandas for data manipulation
import pandas as pd
#import numpy as np for numerical computing
import numpy as np
# import seaborn library for data visualization
import seaborn as sns
# import matplotlib library for creating plots
import matplotlib.pyplot as plt
# import warnings library to manage warning messages
import warnings
# ignore all warning messages
warnings.filterwarnings("ignore")

# Define color variables
header_color = "#c24d2c"  # red color
div_color = "#feffe0"  # yellow color
subheader_color = "#000"  # yellow color

# set the title of the Streamlit app
st.markdown(f"<h1 style='color: {header_color};'>Project 2. Classification Algorithms</h1>", unsafe_allow_html=True)

# add subheader
st.markdown(f"<h2 style='color: {subheader_color};'>Loan Eligibility Prediction model</h2>", unsafe_allow_html=True)
# load the dataset from a CSV file located in the 'data' folder
df = pd.read_csv('data/credit.csv')

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

# read the data
st.markdown(f"<h2 style='color: {header_color};'>Reading the data</h2>", unsafe_allow_html=True)

# # display the first five rows of the dataset in the app
st.write('The dataset is loaded. The first five and last five records displayed below:')
st.write(df.head())
st.write(df.tail())

# create variables for rows and columns counts
rows_count = df.shape[0]
columns_count = df.shape[1]
# display dataset shape
st.markdown(f"""
    <div style='background-color: {div_color}; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
            The dataset contains:
            <ul>
             <li><strong>Rows:</strong> { rows_count }</li>
             <li><strong>Columns:</strong> { columns_count }</li>
            </ul>
    </div>
    <hr>
""", unsafe_allow_html=True)

# create visualization
# count Load_Approved values
st.markdown("##### How many application were approved and how many were denied?")

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



# Count Approved and Disapproved loans
yes_count = df['Loan_Approved'].value_counts().get('Y', 0)
no_count = df['Loan_Approved'].value_counts().get('N', 0)
yes_no_count = yes_count + no_count

# percent_approved = round(yes_count/yes_no_count, 1)*100
percent_approved = yes_count/yes_no_count*100
# round the result to one decimal place
round_percent_approved = round(percent_approved, 1)
# display message

# wrap it all in one div
st.markdown(f"""
<div style='background-color: {div_color}; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
    {yes_count} people (around {round_percent_approved}%) out of {yes_no_count} were approved for loan
</div>
""", unsafe_allow_html=True)

# handle missing values
# display subheader
st.markdown("##### Missing value imputation")

# check for missing values in each variable
st.write("Count missing values in each of the dataset columns:")
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
  
# display subheader
st.markdown("##### Methods to fill in the missing values:")

# write the handling of the missing values steps  
st.markdown("""
            - For numerical variables: imputate using mean or median
            - For categorical variables: imputate using mode
            
            For e.g. in the `Loan_Amount_Term` variable, the value of 360 is repeating the most.
            
            Check that by using `train['Loan_Amount_Term'].value_counts()`
            and replace the missing values in this variable using the mode of this variable. i.e. 360
            
            For the `LoanAmount` variable, check if the variable has outliers by plotting a box plot. 
            
            If there are outliers use the median to fill the null values since mean is highly affected 
            by the presence of outliers. If there are no outliers use mean to impute missing values in `LoanAmount`
            """)

# display subheader
st.markdown("##### Check the DataFrame columns' types:")

#create variable
df_types = df.dtypes
# display column data types in a table 
st.dataframe(df_types)

# display subheader
st.markdown("##### Count values of the 'Gender' column:")

#create a variable
gen_col_values = df['Gender'].value_counts()
# display column data types in a table 
st.dataframe(gen_col_values)

# display subheader
st.markdown("##### Find the mode in the 'Dependents' column")

# find the most frequent value (mode) in the 'Dependents' column
# [0] selects the first mode in case there are multiple modes
dependents_mode = df['Dependents'].mode()[0]
st.write(f"The mode in 'Dependents' column: `{dependents_mode}`")

# display subheader
st.markdown("##### The LoanAmount distribution plot:")
# plot the distribution (histogram + curve) of the 'LoanAmount' column
# Using seaborn's distplot to visualize how loan amounts are spread across the dataset
# Create a figure
fig, ax = plt.subplots(figsize=(8, 3))

# Create the distribution plot on that figure
sns.distplot(df['LoanAmount'], ax=ax)

# Display the plot in Streamlit
st.pyplot(fig)



# display subheader
st.markdown("##### Converting 'Credit_History' and 'Loan_Amount_Term' to object type")

# convert columns to object type
# credit_history_obj_type = df['Credit_History'] = df['Credit_History'].astype('object')
# loan_amount_term_obj_type = df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('object')
df['Credit_History'] = df['Credit_History'].astype('object')
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('object')
df_types = df.dtypes
st.dataframe(df_types)

# display subheader
st.markdown("##### Finding 'Married' column mode")
# find the most frequent value (mode) in the 'Married' column
# [0] selects the first mode in case there are multiple modes
married_mode = df['Married'].mode()[0]
st.write(f"The Mode 'Married': `{married_mode}`")



# display subheader
st.markdown("##### Handling Missing Values")

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

# Confirm if there are any missing values left
confirm_missing_values = df.isnull().sum()
st.write("Confirm if there are any missing values left:")
st.dataframe(confirm_missing_values)



# data prep section
st.markdown(f"<h2 style='color: {header_color};'>Data Preparation</h2>", unsafe_allow_html=True)
st.write("The `Loan_ID` variable will be dropped from the data as it isn't needed for analysis")
st.write("Drop 'Loan_ID' variable from the data and display the first three rows of the dataset")

# drop 'Loan_ID' variable from the data. We won't need it.
df = df.drop('Loan_ID', axis=1)

df_head_3 = df.head(3)
st.dataframe(df_head_3)

# display subheader
st.write("Check data types:")
df_types = df.dtypes
st.dataframe(df_types)

# create a copy of the original DataFrame and store it in 'raw'
# this is useful to keep the original data unchanged for backup or reference
raw = df.copy()

# generate descriptive statistics for all columns, including both numerical and categorical variables
# 'include="all"' ensures that it summarizes ALL columns, not just numeric ones
describe = df.describe(include='all')

# display subheader
st.write("Descriptive statistics for all columns:")
st.dataframe(describe)

# display subheader
st.write("Display the first two rows of the dataset:")
df_head_2 = df.head(2)
st.dataframe(df_head_2)

# display subheader
st.write("Create dummy variables for all 'object' type variables except `Loan_Status` and display first two rows of the dataset:")
# Create dummy variables for all 'object' type variables except 'Loan_Status'
df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents','Education','Self_Employed','Property_Area'], dtype=int)
df_head_2 = df.head(2)
st.dataframe(df_head_2)

# replace values in Loan_approved column
df['Loan_Approved'] = df['Loan_Approved'].replace({'Y':1, 'N':0})
st.write("Replace values in `Loan_Approved` column `Y -> 1` and `N -> 0` and display the first two rows:")
df_head_2 = df.head(2)
st.dataframe(df_head_2)

# saving this processed dataset
file_name = 'Processed_Credit_Dataset.csv'
df.to_csv(f"data/{file_name}", index=None)
st.write(f"saving this processed dataset: `data/{file_name}`")

# add Data Partition section
# wrote a subheader
st.markdown(f"<h2 style='color: {header_color};'>Data Partition</h2>", unsafe_allow_html=True)

# Seperate the input features and target variable
x = df.drop('Loan_Approved',axis=1)
y = df.Loan_Approved

# splitting the data in training and testing set
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, stratify=y)

# create variables for rows and columns counts
rows_count = df.shape[0]
columns_count = df.shape[1]
# display dataset shape
st.markdown(f"""
    <div style='background-color: {div_color}; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
            The dataset contains:
            <ul>
             <li><strong>Rows:</strong> { rows_count }</li>
             <li><strong>Columns:</strong> { columns_count }</li>
            </ul>
    </div>
    <hr>
""", unsafe_allow_html=True)

xtrain_shape, xtest_shape, ytrain_shape, ytest_shape = xtrain.shape, xtest.shape, ytrain.shape, ytest.shape
st.markdown(f"""
The splitted data dimensions in training and testing set:

- `x-train`: {xtrain_shape}
- `x-test`: {xtest_shape}
- `y-train`: {ytrain_shape}
- `y-test`: {ytest_shape}
""")

# display subheader
st.markdown("##### Feature Scaling Using Min-Max Scaler")

# import the MinMaxScaler from scikit-learn
from sklearn.preprocessing import MinMaxScaler

# initialize the scaler
scale = MinMaxScaler()

# display the first two rows of the training data to understand the raw input before scaling
xtrain_head_2 = xtrain.head(2)
st.write("Display the first two rows of the training data to understand the raw input before scaling")
st.dataframe(xtrain_head_2)

# scale the training and testing data
# first, fit the scaler on the training data and transform it
xtrain_scaled = scale.fit_transform(xtrain)

# then, transform the testing data using the same scaler (important: do NOT fit again on test data)
xtest_scaled = scale.transform(xtest)

# add a subheader
st.markdown(f"<h2 style='color: {subheader_color};'>Models</h2>", unsafe_allow_html=True)

# add a subheader
st.markdown(f"<h3 style='color: {header_color};'>1. Logistic Regression</h3>", unsafe_allow_html=True)

# logistic Regression Model and Accuracy Evaluation

# import the LogisticRegression model from scikit-learn
from sklearn.linear_model import LogisticRegression
# train a Logistic Regression model on the scaled training data
lrmodel = LogisticRegression().fit(xtrain_scaled, ytrain)

# Predict the loan eligibility on testing set and calculate its accuracy.
# First, from sklearn.metrics import accuracy_score and confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix

# predict the loan eligibility on the scaled testing data
ypred = lrmodel.predict(xtest_scaled)

# calculate the accuracy score by comparing the predictions to the true labels
accuracy_score = accuracy_score(ypred, ytest)
# write subheader
st.markdown("##### Logistic Regression Model and Accuracy Evaluation")
st.write("Calculate the accuracy score by comparing the predictions to the true labels")
# display the accuracy score in the Streamlit app
st.write(accuracy_score)

# create a DataFrame to compare actual vs predicted values
comparison_df = pd.DataFrame({
    'Actual': ytest,
    'Predicted': ypred
})

# display the comparison table
st.subheader("Actual vs Predicted Loan Eligibility")
st.dataframe(comparison_df)

# display the confusion matrix
conf_matrix = confusion_matrix(ytest, ypred)
st.write("Confusion Matrix")
st.dataframe(conf_matrix)


# import accuracy_score
from sklearn.metrics import accuracy_score

# predict probabilities
pypred = lrmodel.predict_proba(xtest_scaled)

# turn probabilities into a DataFrame
proba_df = pd.DataFrame(pypred, columns=["Probability_No", "Probability_Yes"])

# display the probability table
st.write("Check how probabilities are assigned")
st.dataframe(proba_df)

# change the threshold to 70%
proba_pred = (pypred[:, 1] >= 0.7).astype(int)

# calculate new accuracy
from sklearn.metrics import accuracy_score
accuracy_proba_pred = accuracy_score(ytest, proba_pred)

st.write("Change the default threshold to 70% and above")
st.write("Count accuracy score")
st.write(accuracy_proba_pred)

# add a subheader
st.markdown(f"<h3 style='color: {header_color};'>2. Random Forest</h3>", unsafe_allow_html=True)

# import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# list the tunable hyperparameters for Random Forest algorithm
list_tunable_hyperparameters = RandomForestClassifier().get_params()
# convert the dictionary to a DataFrame
hyperparameters_df = pd.DataFrame(list(list_tunable_hyperparameters.items()), columns=["Hyperparameter", "Default Value"])

# display the dataframe
st.markdown("##### Tunable Hyperparameters for Random Forest")
st.dataframe(hyperparameters_df)

st.markdown("""
            For random forests,
               - The first hyperparameter to tune is n_estimators. We will try 100 and 200.
               - The second one is max_features. Let's try - 'auto', 'sqrt', and 0.33.
               - The third one is Max_depth. Let's try - 3, 4
            """)

# create variables for rows and columns counts
rows_count = xtrain.shape[0]
columns_count = xtrain.shape[1]
# display dataset shape
st.markdown(f"""
    <div style='background-color: {div_color}; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
            x-train shape:
            <ul>
             <li><strong>Rows:</strong> { rows_count }</li>
             <li><strong>Columns:</strong> { columns_count }</li>
            </ul>
    </div>
    <hr>
""", unsafe_allow_html=True)

# display a subheader
st.markdown("##### Hyperparameter Tuning")

# import the RandomForestClassifier from scikit-learn
from sklearn.ensemble import RandomForestClassifier

# create a Random Forest model with specific hyperparameters:
# n_estimators=2 -> number of trees in the forest
# max_depth=2 -> maximum depth of each tree
# max_features=10 -> number of features to consider when looking for the best split
rfmodel = RandomForestClassifier(n_estimators=2,
                                  max_depth=2,
                                  max_features=10)

# train (fit) the Random Forest model on the training data
rfmodel.fit(xtrain, ytrain)

# predict the target values (loan approval) on the testing data
ypred = rfmodel.predict(xtest)

# import metrics to evaluate the model
# from sklearn.metrics import accuracy_score, confusion_matrix

# display the accuracy score (how many correct predictions)
st.write(accuracy_score(ytest, ypred), '\n')

# compute the confusion matrix
conf_matrix = confusion_matrix(ytest, ypred)

# turn it into a labeled DataFrame
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=["Actual No", "Actual Yes"],      # row labels
    columns=["Predicted No", "Predicted Yes"]  # column labels
)

# display the confusion matrix
st.markdown("##### Confusion Matrix")
st.dataframe(conf_matrix_df)

# get feature importances
feature_importance = rfmodel.feature_importances_

# create a DataFrame: feature names + importance scores
importance_df = pd.DataFrame({
    'Feature': xtrain.columns,
    'Importance': feature_importance
})

# sort features by importance (high to low)
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# display with highlight for Top 3
st.markdown("##### Feature Importance Table (Top 3 Highlighted)")

# function to highlight top 3 rows
def highlight_top_3(s):
    return ['background-color: #ffeb99' if i < 3 else '' for i in range(len(s))]

# apply highlighting
styled_df = importance_df.style.apply(highlight_top_3, axis=0)

# show the styled table
st.dataframe(styled_df)

# plot as a bar chart
st.markdown("##### Feature Importance Chart")
fig, ax = plt.subplots()
ax.barh(importance_df['Feature'], importance_df['Importance'])
ax.invert_yaxis()  # top feature on top
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance')
st.pyplot(fig)

# add a subheader
st.markdown(f"<h3 style='color: {header_color};'>Cross Validation</h3>", unsafe_allow_html=True)

# import rquired libraries
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# if you have a imbalanced dataset, you can use stratifiedKFold
from sklearn.model_selection import StratifiedKFold

# display a subheader
st.markdown("##### Cross-Validation Results for Logistic Regression")

# set up a KFold cross-validation
kfold = KFold(n_splits=5)

# use cross-validation to evaluate the model
lr_scores = cross_val_score(lrmodel, xtrain_scaled, ytrain, cv=kfold)

# display the accuracy scores for each fold as a dataframe
st.markdown("##### Accuracy Scores for Each Fold:")
st.dataframe(pd.DataFrame(lr_scores, columns=["Accuracy Score"]))

# display the mean accuracy and standard deviation nicely
st.markdown("##### Summary Statistics:")
st.markdown(f"- **Mean Accuracy:** `{lr_scores.mean():.4f}`")
st.markdown(f"- **Standard Deviation:** `{lr_scores.std():.4f}`")

# display a subheader
st.markdown("##### Cross-Validation Results for Random Forest")

# set up a KFold cross-validation
kfold = KFold(n_splits=5)

# use cross-validation to evaluate the model
rf_scores = cross_val_score(rfmodel, xtrain_scaled, ytrain, cv=kfold)

# display the accuracy scores for each fold as a dataframe
st.markdown("##### Accuracy Scores for Each Fold:")
st.dataframe(pd.DataFrame(rf_scores, columns=["Accuracy Score"]))

# display the mean accuracy and standard deviation nicely
st.markdown("##### Summary Statistics:")
st.markdown(f"- **Mean Accuracy:** `{rf_scores.mean():.4f}`")
st.markdown(f"- **Standard Deviation:** `{rf_scores.std():.4f}`")



