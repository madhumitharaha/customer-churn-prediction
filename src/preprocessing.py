import pandas as pd

def preprocess_data(df):

    # Drop ID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric (VERY IMPORTANT)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Fill missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Separate categorical & numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # One-hot encode ONLY categorical columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df
