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







selected_machine_downtime = monthly_downtime_all_machines[monthly_downtime_all_machines['Machine_ID'] == machine]
st.write(selected_machine_downtime[['Month', 'Downtime_Percentage']])






df['Month'] = df['Date'].dt.to_period('M')

# Group by machine and month, and count occurrences of 'Machine_Failure' and 'No_Machine_Failure'
monthly_downtime_all_machines = df.groupby(['Machine_ID', 'Month'])['Downtime'].value_counts().unstack(fill_value=0)

# Rename the columns for clarity
monthly_downtime_all_machines.columns = ['No Machine Failure', 'Machine Failure']

# Calculate total days in each month and downtime percentage for all machines
monthly_downtime_all_machines['Total_Days'] = monthly_downtime_all_machines['No Machine Failure'] + monthly_downtime_all_machines['Machine Failure']
monthly_downtime_all_machines['Downtime_Percentage'] = (monthly_downtime_all_machines['Machine Failure'] / monthly_downtime_all_machines['Total_Days']) * 100

# Reset index to flatten the MultiIndex and make 'Month' a column
monthly_downtime_all_machines = monthly_downtime_all_machines.reset_index()

# Convert 'Month' to string for proper x-axis labels in the bar chart
monthly_downtime_all_machines['Month'] = monthly_downtime_all_machines['Month'].astype(str)

# Exclude the first and last month based on the 'Month' column
min_month = monthly_downtime_all_machines['Month'].min()
max_month = monthly_downtime_all_machines['Month'].max()

# Filter out the first and last month
monthly_downtime_all_machines = monthly_downtime_all_machines[(monthly_downtime_all_machines['Month'] > min_month) & (monthly_downtime_all_machines['Month'] < max_month)]

# Set the title for the layout
st.title(f"Downtime Comparison for Machine {machine} vs Others")

# Create the bar chart showing monthly downtime proportions for all machines
fig = px.bar(monthly_downtime_all_machines,
             x='Month',  # X-axis: months
             y='Downtime_Percentage',  # Y-axis: downtime percentage
             color='Machine_ID',  # Color bars by machine
             labels={'Downtime_Percentage': 'Downtime Percentage (%)', 'Month': 'Month', 'Machine_ID': ' '},
             barmode='group')

# Display the chart
st.plotly_chart(fig)
