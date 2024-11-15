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

# Group by machine and month, and count occurrences of 'Machine_Failure' and 'No_Machine_Failure'
monthly_downtime_all_machines = df.groupby(['Machine_ID', 'Month'])['Downtime'].value_counts().unstack(fill_value=0)

# Rename the columns for clarity
monthly_downtime_all_machines.columns = ['No Machine Failure', 'Machine Failure']

# Calculate total days in each month and downtime percentage for all machines
monthly_downtime_all_machines['Total_Days'] = monthly_downtime_all_machines['No Machine Failure'] + monthly_downtime_all_machines['Machine Failure']
monthly_downtime_all_machines['Downtime_Percentage'] = (monthly_downtime_all_machines['Machine Failure'] / monthly_downtime_all_machines['Total_Days']) * 100

# Exclude the first and last month
monthly_downtime_all_machines = monthly_downtime_all_machines.iloc[1:-1]

# Create a container to hold both the graph and table
container = st.container()

# Set a single overarching title for the entire layout
container.title(f"Downtime Comparison for Machine {machine} vs Others")

# Create columns for layout
col1, col2 = container.columns([2, 1])  # Graph takes up 2/3 and table takes up 1/3

# In the first column (larger), show the bar chart comparing downtime proportions
with col1:
    # Create a comparison bar chart for all machines
    fig = px.bar(monthly_downtime_all_machines,
                 x=monthly_downtime_all_machines.index.astype(str), 
                 y='Downtime_Percentage', 
                 color=monthly_downtime_all_machines.index.get_level_values('Machine_ID'),
                 title=f"Comparison of Monthly Downtime Proportions",
                 labels={'Downtime_Percentage': 'Downtime Percentage (%)', 'Month': 'Month', 'Machine_ID': 'Machine'},
                 barmode='group')
    st.plotly_chart(fig)

# In the second column (smaller), show the table for the selected machine
with col2:
    # Filter for the selected machine and show the monthly downtime proportions
    selected_machine_downtime = monthly_downtime_all_machines.xs(machine, level='Machine_ID')
    st.write(selected_machine_downtime[['Downtime_Percentage']])
