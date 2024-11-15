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











df_cleaned = df.dropna(subset=['Downtime'])

# Encode the categorical target variable 'Downtime' (if necessary)
label_encoder = LabelEncoder()
df_cleaned['Downtime'] = label_encoder.fit_transform(df_cleaned['Downtime'])

# Features and target
X = df_cleaned[['Temperature', 'Pressure', 'Operational_Time']]  # Add other relevant features here
y = df_cleaned['Downtime']

# Split the data for training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
feature_importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame to display the critical variables and their importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display most critical variables
st.markdown(f"<h3 style='text-align: center;'>Critical Variables Causing Machine Failure</h3>", unsafe_allow_html=True)
st.table(importance_df)

# --------------- HEALTH SCORE -------------------

# Get the predicted probabilities for the machine's failure
machine_data = machine_data.sort_values(by='Date')
X_machine = machine_data[['Temperature', 'Pressure', 'Operational_Time']]  # Adjust features as needed

# Ensure that we are predicting for the selected machine
if not X_machine.empty:
    health_probabilities = model.predict_proba(X_machine)[:, 1]  # Probability of failure (class 1)
    health_scores = 1 - health_probabilities  # Higher probability means worse health, so inverse for health score

    # Add health score to the machine data
    machine_data['Health_Score'] = health_scores

    # Display health score graph
    st.markdown(f"<h3 style='text-align: center;'>Health Score of the Machine</h3>", unsafe_allow_html=True)
    fig_health = px.line(machine_data, 
                         x='Date', 
                         y='Health_Score', 
                         labels={'Date': 'Date', 'Health_Score': 'Health Score'}, 
                         title=f"Health Score of Machine {machine}")
    st.plotly_chart(fig_health)

else:
    st.warning("Not enough data available for health score calculation.")



