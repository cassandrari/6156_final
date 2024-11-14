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








downtime_trends = machine_data.groupby('Date')['Machine_Failure'].value_counts().unstack(fill_value=0)

# The 'Machine_Failure' and 'No_Machine_Failure' are values, not column names
downtime_trends.columns = ['No Machine Failure', 'Machine Failure']  # Rename the columns

# Calculate Monthly Downtime
machine_data['Month'] = machine_data['Date'].dt.to_period('M')  # Extract month-year from date
monthly_downtime = machine_data.groupby('Month')['Machine_Failure'].value_counts().unstack(fill_value=0)

# Rename the columns for clarity
monthly_downtime.columns = ['No Machine Failure', 'Machine Failure']

# Calculate the percentage of downtime for each month
monthly_downtime['Total_Days'] = monthly_downtime['No Machine Failure'] + monthly_downtime['Machine Failure']
monthly_downtime['Downtime_Percentage'] = (monthly_downtime['Machine Failure'] / monthly_downtime['Total_Days']) * 100

# Identify months where downtime is greater than 10%
major_downtime_months = monthly_downtime[monthly_downtime['Downtime_Percentage'] > 10]

# Create the line chart for daily downtime trend
st.subheader(f"Downtime Trend for {machine}")

# Create Plotly figure for downtime trend
fig = px.line(downtime_trends, 
              x=downtime_trends.index, 
              y='Machine Failure', 
              title=f'Downtime Trend for {machine}', 
              labels={'Date': 'Date', 'Machine Failure': 'Machine Failure Events'})

# Highlight months where downtime is greater than 10% with vertical rectangles
for month in major_downtime_months.index:
    # Find the first and last day of the month
    start_date = month.start_time
    end_date = month.end_time

    # Add vertical rectangles to highlight the month
    fig.add_vrect(x0=start_date, 
                  x1=end_date, 
                  fillcolor="red", 
                  opacity=0.2, 
                  line_width=0)

# Display the plot
st.plotly_chart(fig)
