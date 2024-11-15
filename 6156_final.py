import streamlit as st
import pandas as pd
import plotly.express as px
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
st.subheader(f"Downtime Trend for {machine}")
downtime_trends = machine_data.groupby('Date')['Downtime'].value_counts().unstack(fill_value=0)
downtime_trends.columns = ['No Downtime', 'Downtime']  # Rename columns for clarity
fig = px.line(downtime_trends, 
              x=downtime_trends.index, 
              y='Downtime', 
              title=f'Downtime Trend for {machine}', 
              labels={'Date': 'Date', 'Downtime': 'Downtime Events'})
st.plotly_chart(fig)









machine_data['Month'] = machine_data['Date'].dt.to_period('M')

# Group by month and count occurrences of 'Machine_Failure' and 'No_Machine_Failure'
monthly_downtime = machine_data.groupby('Month')['Downtime'].value_counts().unstack(fill_value=0)

# Rename the columns for clarity
monthly_downtime.columns = ['No Machine Failure', 'Machine Failure']

# Calculate total days in each month and downtime percentage
monthly_downtime['Total_Days'] = monthly_downtime['No Machine Failure'] + monthly_downtime['Machine Failure']
monthly_downtime['Downtime_Percentage'] = (monthly_downtime['Machine Failure'] / monthly_downtime['Total_Days']) * 100

# Exclude the first and last month
monthly_downtime = monthly_downtime.iloc[1:-1]

# Create a container to hold both the graph and table
container = st.container()

# Set a single overarching title for the entire layout
container.title(f"Downtime Analysis for Machine {machine}")

# Create columns for layout
col1, col2 = container.columns([2, 1])  # Graph takes up 2/3 and table takes up 1/3

# In the first column (larger), show the line chart
with col1:
    # Group by Date and count failure events ('Machine_Failure' and 'No_Machine_Failure')
    downtime_trends = machine_data.groupby('Date')['Downtime'].value_counts().unstack(fill_value=0)
    downtime_trends.columns = ['No Machine Failure', 'Machine Failure']  # Rename the columns for clarity

    # Create the line chart for downtime trend
    fig = px.line(downtime_trends, 
                  x=downtime_trends.index, 
                  y='Machine Failure',  
                  labels={'Date': 'Date', 'Machine Failure': 'Machine Failure Events'})
    st.plotly_chart(fig)

# In the second column (smaller), show the table
with col2:
    st.write(monthly_downtime[['Downtime_Percentage']])
