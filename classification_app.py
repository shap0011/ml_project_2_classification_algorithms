# Loading all the necessary packages

import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'info'

import logging

# Configure logging manually again
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
# import functions from app_module
from app_module import functions as func

warnings.filterwarnings('ignore')

try: 
    
    #-------- page setting, header, intro ---------------------
       
    # Set page configuration
    st.set_page_config(page_title="Loan Eligibility App", layout="wide")
    
    # Define color variables
    header_color = "#c24d2c"  # red color
    div_color = "#feffe0"  # yellow color
    subheader_color = "#000"  # yellow color

    # set the title of the Streamlit app
    # st.markdown(f"<h1 style='color: {header_color};'>Project 2. Classification Algorithms</h1>", unsafe_allow_html=True)

    # add subheader
    # st.markdown(f"<h2 style='color: {subheader_color};'>Loan Eligibility Prediction model</h2>", unsafe_allow_html=True)
    # load the dataset from a CSV file located in the 'data' folder
    
    #-------- the app overview -----------------------------
    
    
    
    #-------- user instructions -------------------------------
    
    
    
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

    # show scope and dictionary
    # func.show_scope(header_color)
    # func.show_data_dictionary(header_color)

    # display the first five rows of the dataset in the app
    # func.preview_data(df, div_color)
    
     #-------- loan approval distribution -----------------------------

    # st.markdown("##### How many application were approved and how many were denied?")

    # create visualization
    # plot loan approval distribution
    func.plot_loan_approval_distribution(df)

    # display percentage approved
    func.display_approval_percentage(df, div_color)

    # handle missing values
    # display subheader
    st.markdown("##### Missing value imputation")

    # check for missing values in each variable
    st.write("Count missing values in each of the dataset columns:")

    # Handle missing values
    func.show_missing_values(df, div_color)
    
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

    # Fill missing values
    df = func.fill_missing_values(df)

    # Confirm if there are any missing values left
    confirm_missing_values = df.isnull().sum()
    st.write("Confirm if there are any missing values left:")
    st.dataframe(confirm_missing_values)



    # data prep section
    st.markdown(f"<h2 style='color: {header_color};'>Data Preparation</h2>", unsafe_allow_html=True)
    st.write("The `Loan_ID` variable will be dropped from the data as it isn't needed for analysis")
    st.write("Drop 'Loan_ID' variable from the data and display the first three rows of the dataset")

    df_head_3 = df.head(3)
    st.dataframe(df_head_3)

    # display subheader
    st.write("Check data types:")
    df_types = df.dtypes
    st.dataframe(df_types)

    # create a copy of the original DataFrame and store it in 'raw'
    raw = df.copy()

    # generate descriptive statistics for all columns
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

    # Prepare data (drop ID, dummies, target encoding)
    df = func.prepare_data(df)

    st.write("Replace values in `Loan_Approved` column `Y -> 1` and `N -> 0` and display the first two rows:")
    df_head_2 = df.head(2)
    st.dataframe(df_head_2)

    # Save processed data
    func.save_processed_data(df, "data/Processed_Credit_Dataset.csv")

    # add Data Partition section
    # wrote a subheader
    st.markdown(f"<h2 style='color: {header_color};'>Data Partition</h2>", unsafe_allow_html=True)

    # splitting the data in training and testing set
    from sklearn.model_selection import train_test_split

    # Split into train/test
    xtrain, xtest, ytrain, ytest = func.split_data(df)

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

    xtrain_shape = str(xtrain.shape)
    xtest_shape = str(xtest.shape)
    ytrain_shape = str(ytrain.shape)
    ytest_shape = str(ytest.shape)

    st.markdown(f"""
    The splitted data dimensions in training and testing set:

    - `x-train`: `{xtrain_shape}`
    - `x-test`: `{xtest_shape}`
    - `y-train`: `{ytrain_shape}`
    - `y-test`: `{ytest_shape}`
    """)


    # display subheader
    st.markdown("##### Feature Scaling Using Min-Max Scaler")

    # import the MinMaxScaler from scikit-learn
    from sklearn.preprocessing import MinMaxScaler

    # display the first two rows of the training data to understand the raw input before scaling
    xtrain_head_2 = xtrain.head(2)
    st.write("Display the first two rows of the training data to understand the raw input before scaling")
    st.dataframe(xtrain_head_2)

    # Scale features
    xtrain_scaled, xtest_scaled = func.scale_features(xtrain, xtest)

    # add a subheader
    st.markdown(f"<h2 style='color: {subheader_color};'>Models</h2>", unsafe_allow_html=True)

    # add a subheader
    st.markdown(f"<h3 style='color: {header_color};'>1. Logistic Regression</h3>", unsafe_allow_html=True)

    # logistic Regression Model and Accuracy Evaluation

    # import the LogisticRegression model from scikit-learn
    from sklearn.linear_model import LogisticRegression

    # First, from sklearn.metrics import accuracy_score and confusion_matrix
    from sklearn.metrics import accuracy_score, confusion_matrix

    # write subheader
    st.markdown("##### Logistic Regression Model and Accuracy Evaluation")
    st.write("Calculate the accuracy score by comparing the predictions to the true labels")

    # display the comparison table
    st.subheader("Actual vs Predicted Loan Eligibility")

    # Logistic Regression
    try:
        lrmodel = func.train_logistic_regression(xtrain_scaled, ytrain)
    except Exception as e:
        logging.error(f"Failed to train Logistic Regression: {e}", exc_info=True)
        st.error("Failed to train Logistic Regression model.")

    try:
        func.evaluate_logistic_regression(lrmodel, xtest_scaled, ytest)
    except Exception as e:
        logging.error(f"Failed to evaluate Logistic Regression: {e}", exc_info=True)
        st.error("Failed to evaluate Logistic Regression model.")


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

    # Random Forest
    rfmodel = func.train_random_forest(xtrain, ytrain)
    func.evaluate_random_forest(rfmodel, xtest, ytest)



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
    st.markdown("##### Cross-Validation Results")

    try:
        func.cross_validate_model(lrmodel, xtrain_scaled, ytrain, "Logistic Regression")
        func.cross_validate_model(rfmodel, xtrain, ytrain, "Random Forest")
    except Exception as e:
        logging.error(f"Cross-validation failed: {e}", exc_info=True)
        st.error("Cross-validation failed. Please check the logs.")

    st.markdown("""
                    Note:<br>
                        <ol>
                            <li>
                                By using cross-validation, we can get a better estimate of the performance 
                                of the model than by using a single train-test split. This is because cross-validation 
                                uses all the data for training and testing, and averages the results over multiple iterations, 
                                which helps to reduce the impact of random variations in the data.
                            </li>
                            <li>
                                `StratifiedKFold` is a variation of KFold that preserves the proportion of samples 
                                for each class in each fold. This is important when the target variable is imbalanced, 
                                i.e., when some classes have many more samples than others. By preserving the class proportions 
                                in each fold, StratifiedKFold ensures that each fold is representative of the overall dataset 
                                and helps to avoid overfitting or underfitting on specific classes.
                            </li>
                        </ol>
                """, unsafe_allow_html=True)
except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    st.error("Something went wrong! Please check the logs or try again later.")


