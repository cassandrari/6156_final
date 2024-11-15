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








df['Month'] = df['Date'].dt.to_period('M')

# Group by machine and month, and count occurrences of 'Machine Failure' and 'No Machine Failure'
monthly_downtime_all_machines = df.groupby(['Machine_ID', 'Month', 'Downtime']).size().unstack(fill_value=0)

# Rename the columns for clarity (after unstack, 'Machine Failure' and 'No Machine Failure' should be the columns)
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

# Now, create a table for the selected machine's downtime proportions across months.
# We only want data for the selected machine.
selected_machine_data = monthly_downtime_all_machines[monthly_downtime_all_machines['Machine_ID'] == machine]

# Create a horizontal table: 'Month' will be columns, downtime proportion will be the row
downtime_proportion_table = selected_machine_data.pivot(index='Machine_ID', columns='Month', values='Downtime_Percentage')

# Remove the 'Machine_ID' column and show only the downtime proportion
downtime_proportion_table = downtime_proportion_table.drop(columns='Machine_ID', errors='ignore')

# Display the table as horizontal with one row for the selected machine
st.subheader(f"Downtime Proportion by Month for {machine}")
st.table(downtime_proportion_table)
