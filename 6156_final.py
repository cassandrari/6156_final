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

# Display the downtime percentage as a table
st.subheader(f"Monthly Downtime Proportion for {machine}")
st.write(monthly_downtime[['Downtime_Percentage']])

# Optionally, you can visualize the downtime percentage as a bar chart
fig = px.bar(monthly_downtime, 
             x=monthly_downtime.index.astype(str), 
             y='Downtime_Percentage', 
             title=f'Monthly Downtime Proportion for {machine}',
             labels={'Downtime_Percentage': 'Downtime (%)', 'Month': 'Month'})
st.plotly_chart(fig)
