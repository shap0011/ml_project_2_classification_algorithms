# Loan Eligibility Prediction App

This project is a **Streamlit web application** that predicts whether a **loan applicant** should be **approved or denied** using **Classification Algorithms**.

The app is designed to:

- **Explore and visualize** the dataset.

- **Preprocess** the data (handle missing values, encode features).

- **Train and evaluate** two Machine Learning models:

    - Logistic Regression

    - Random Forest Classifier

- **Tune hyperparameters** and **cross-validate** models.

- **Predict loan eligibility** with **target accuracy > 76%**.

## Project Structure

- **.streamlit/**
  - `config.toml` — Theme setting
- `classification_app.py` — Main Streamlit app
- **app_module/**
  - `__init__.py`
  - `functions.py` — All helper functions
- **data/**
  - `credit.csv` — Raw dataset
  - `Processed_Credit__Dataset.csv` — Processed dataset
- `requirements.txt` — List of Python dependencies
- `README.md` — Project documentation

## Dataset
- **Source:** German Credit Dataset

- **Rows:** 614 loan applications

- **Columns:** 13 features (Gender, Education, Applicant Income, Loan Amount, etc.)

- **Target Variable:** Loan_Approved (Yes/No)

## Models Used

| Model                    | Description                        |
|--------------------------|------------------------------------|
| Logistic Regression      | Baseline classification model      |
| Random Forest Classifier | Tree-based regression model        |


- **Evaluation Metric:** Mean Accuracy and Confusion Matrix

- **Cross-Validation:** 5-fold K-Fold Cross Validation used

## Logging and Error Handling

Backend logging is enabled using Python’s logging module.

Important messages are displayed inside the app using Streamlit's st.success(), st.warning(), st.error().

The app is robust against missing files or bad data.

## How to Run the App Locally

1. **Clone the repository**

```bash```
git clone https://github.com/shap0011/ml_project_2_classification_algorithms.git
cd ml_project_2_classification_algorithms

2. **Install the required packages**

```bash```
    pip install -r requirements.txt

3. **Run the App**

```bash```
streamlit run classification_app.py

4. Open the URL shown (usually http://localhost:8501) to view the app in your browser!

## Deployment
The app is also deployed on Streamlit Cloud.
Click [![Here](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-project-2-classification-algorithms.streamlit.app/) to view the live app.

## Author
Name: Olga Durham

LinkedIn: [\[Olga Durham LinkedIn Link\]](https://www.linkedin.com/in/olga-durham/)

GitHub: [\[Olga Durham GitHub Link\]](https://github.com/shap0011)

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://cuddly-xylophone-4vq4xxxjwqjf75vj.github.dev/)
