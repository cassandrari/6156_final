import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Streamlit App Title
st.title('Machine Downtime Prediction')

# Upload CSV File
st.sidebar.header('Upload Your CSV File')
uploaded_file = pd.read_csv('Machine_Downtime.csv')

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Display basic information about the dataset
    st.write("Dataset Overview")
    st.write(df.head())
    st.write(f"Number of records: {df.shape[0]}")

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Drop rows where 'Date' is NaT (if any)
    df.dropna(subset=['Date'], inplace=True)

    # Visualize the distribution of numerical features
    st.subheader('Distribution of Numerical Features')
    numerical_features = df.select_dtypes(include=[np.number]).columns

    # Display histograms for numerical features
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    axes = axes.flatten()
    for i, feature in enumerate(numerical_features):
        sns.histplot(df[feature], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}')
    plt.tight_layout()
    st.pyplot(fig)

    # Correlation Heatmap: Visualize correlation between numerical features
    st.subheader('Correlation Matrix Heatmap')
    correlation_matrix = df[numerical_features].corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Preprocessing: Label Encoding for 'Downtime' column
    st.subheader('Label Encoding for Downtime')
    st.write(f"Unique values in 'Downtime' before encoding: {df['Downtime'].unique()}")
    label_encoder = LabelEncoder()
    df['Downtime'] = label_encoder.fit_transform(df['Downtime'])
    st.write(f"Unique values in 'Downtime' after encoding: {df['Downtime'].unique()}")

    # Model Training
    st.subheader('Model Training and Evaluation')

    # Define features (X) and target variable (y)
    X = df[['Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)', 'Air_System_Pressure(bar)', 
            'Coolant_Temperature', 'Hydraulic_Oil_Temperature(?C)', 'Spindle_Bearing_Temperature(?C)',
            'Spindle_Vibration(?m)', 'Tool_Vibration(?m)', 'Spindle_Speed(RPM)', 'Voltage(volts)', 
            'Torque(Nm)', 'Cutting(kN)', 'Rolling_Hydraulic_Temperature']]

    # Target variable: Downtime (now numeric)
    y = df['Downtime']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display evaluation metrics
    st.write(f'Mean Squared Error (MSE): {mse:.4f}')
    st.write(f'R-squared: {r2:.4f}')

    # Convert continuous predictions to class labels (0 or 1)
    y_pred_class = (y_pred > 0.5).astype(int)

    # Calculate Precision, Recall, F1-Score
    precision = precision_score(y_test, y_pred_class)
    recall = recall_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_class)

    # Display classification metrics
    st.write(f'Precision: {precision:.4f}')
    st.write(f'Recall: {recall:.4f}')
    st.write(f'F1-Score: {f1:.4f}')
    st.write('Confusion Matrix:')
    st.write(cm)

    # Display confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No_Machine_Failure', 'Machine_Failure'], yticklabels=['No_Machine_Failure', 'Machine_Failure'], ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)
else:
    st.write("Please upload a CSV file to get started.")
