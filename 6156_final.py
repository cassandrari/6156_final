import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#dataset
df = pd.read_csv('Machine_Downtime.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] >= '2021-12-20']
machines = df['Machine_ID'].unique()

#select box
machine = st.selectbox("Select Machine", machines)
machine_data = df[df['Machine_ID'] == machine]





#trend analysis
machine_data = machine_data.sort_values(by='Date')
st.markdown(f"<h3 style='text-align: center;'>Downtime Trends by Month</h3>", unsafe_allow_html=True)
downtime_trends = machine_data.groupby('Date')['Downtime'].value_counts().unstack(fill_value=0)
downtime_trends.columns = ['No Downtime', 'Downtime']  # Rename columns for clarity
fig = px.line(downtime_trends, 
              x=downtime_trends.index, 
              y='Downtime', 
              labels={'Date': 'Date', 'Downtime': 'Downtime Events'})
st.plotly_chart(fig)






#propotion by month table
df['Month'] = df['Date'].dt.to_period('M')
monthly_downtime_all_machines = df.groupby(['Machine_ID', 'Month', 'Downtime']).size().unstack(fill_value=0)
monthly_downtime_all_machines.columns = ['No Machine Failure', 'Machine Failure']
monthly_downtime_all_machines['Total_Days'] = monthly_downtime_all_machines['No Machine Failure'] + monthly_downtime_all_machines['Machine Failure']
monthly_downtime_all_machines['Downtime_Percentage'] = (monthly_downtime_all_machines['Machine Failure'] / monthly_downtime_all_machines['Total_Days']) * 100
monthly_downtime_all_machines = monthly_downtime_all_machines.reset_index()
monthly_downtime_all_machines['Month'] = monthly_downtime_all_machines['Month'].astype(str)
min_month = monthly_downtime_all_machines['Month'].min()
max_month = monthly_downtime_all_machines['Month'].max()
monthly_downtime_all_machines = monthly_downtime_all_machines[(monthly_downtime_all_machines['Month'] > min_month) & (monthly_downtime_all_machines['Month'] < max_month)]
selected_machine_data = monthly_downtime_all_machines[monthly_downtime_all_machines['Machine_ID'] == machine]
downtime_proportion_table = selected_machine_data.pivot(index='Machine_ID', columns='Month', values='Downtime_Percentage')
downtime_proportion_table = downtime_proportion_table.drop(columns='Machine_ID', errors='ignore')
st.markdown(f"<h3 style='text-align: center;'>Downtime Proportion by Month</h3>", unsafe_allow_html=True)
st.table(downtime_proportion_table)







if df['Downtime'].dtype == 'object':  # Only apply LabelEncoder if it's not numeric
    if df['Downtime'].isna().sum() > 0:
        # Handle missing values in Downtime column before encoding (e.g., by filling with a default value)
        df['Downtime'] = df['Downtime'].fillna('No Machine Failure')
    le = LabelEncoder()
    df['Downtime'] = le.fit_transform(df['Downtime'])  # 'Machine Failure' -> 1, 'No Machine Failure' -> 0

# Dynamically select all columns except 'Machine_ID', 'Downtime', 'Date', 'Assembly_Line_No' for prediction
features = [col for col in df.columns if col not in ['Machine_ID', 'Downtime', 'Date', 'Assembly_Line_No']]

# Handle missing values by filling with the column's mean (other strategies like median or mode may also work)
# Select only numeric columns and fill missing values in these columns
numeric_features = df[features].select_dtypes(include=['float64', 'int64'])
df[numeric_features.columns] = numeric_features.fillna(numeric_features.mean())

# Prepare feature matrix (X) and target vector (y)
X = df[features]
y = df['Downtime']

# Ensure no infinite values in the feature matrix
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())

# Train a Random Forest Classifier to assess feature importance
model = RandomForestClassifier(n_estimators=100, random_state=42)

try:
    model.fit(X, y)
except ValueError as e:
    st.error(f"Error during model fitting: {e}")
else:
    # Get the feature importance from the model
    feature_importance = model.feature_importances_

    # Create a DataFrame to show the feature importance in a readable format
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    })

    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Streamlit Display: Show a title
    st.markdown("<h3 style='text-align: center;'>Top Predictors of Machine Failure</h3>", unsafe_allow_html=True)

    # Display feature importance table
    st.write("### Key Machine Metrics That Impact Failure Prediction")
    st.table(feature_importance_df)

    # Visualize the top 2-3 important features with a bar chart
    top_features = feature_importance_df.head(3)

    fig = plt.figure(figsize=(8, 5))
    plt.bar(top_features['Feature'], top_features['Importance'], color='skyblue')
    plt.title('Top 3 Features Affecting Machine Failure', fontsize=14)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    st.pyplot(fig)

    # Now you can display additional details about these metrics
    # Example: Show summary statistics for the most important variables
    st.write("### Summary Statistics of Important Variables")

    for feature in top_features['Feature']:
        st.write(f"**{feature}**:")
        st.write(df[feature].describe())
