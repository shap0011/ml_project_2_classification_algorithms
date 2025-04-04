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

# set the title of the Streamlit app
st.markdown("<h1 style='color: #c24d2c;'>Project 2. Classification Algorithms</h1>", unsafe_allow_html=True)

# add subheader
st.subheader("Loan Eligibility Prediction model")
# load the dataset from a CSV file located in the 'data' folder
df = pd.read_csv('data/credit.csv')

# add the project scope and description
st.markdown("#### Project Scope:")
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
st.markdown("#### Data Dictionary:")
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
st.markdown("#### Reading the data")

# # display the first five rows of the dataset in the app
st.write('The dataset is loaded. The first five and last five records displayed below:')
st.write(df.head())
st.write(df.tail())

# create variables for rows and columns counts
rows_count = df.shape[0]
columns_count = df.shape[1]
# display dataset shape
st.markdown(f"""
            The dataset contains:
             - **Rows:** { rows_count }
             - **Columns:** { columns_count }
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
st.write(f"{yes_count} people (around {round_percent_approved}%) out of {yes_no_count} were approved for loan")

# handle missing values
# display subheader
st.markdown("##### Missing value imputation")

# check for missing values in each variable
st.write("Count missing values in each of the dataset columns:")
missing_values = df.isnull().sum()
# display dataframe table
# st.dataframe(missing_values)

# loop through the missing values dictionary and print column names and their missing values side by side
# for column, missing_count in missing_values.items():
#     st.markdown(f"<span style='background-color: #feffdf;'>**{column}:** {missing_count} missing</span>", unsafe_allow_html=True)

# build HTML string
all_missing = ""

for column, missing_count in missing_values.items():
    all_missing += f"<strong>{column}:</strong> {missing_count} missing<br>"

# wrap it all in one div
st.markdown(f"""
<div style='background-color: #feffdf; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
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
            
            For the `LoanAmoun`t variable, check if the variable has ouliers by plotting a box plot. 
            
            If there are outliers use the median to fill the null values since mean is highly affected 
            by the presence of outliers. If there are no outliers use mean to impute missing values in `LoanAmount`
            """)



