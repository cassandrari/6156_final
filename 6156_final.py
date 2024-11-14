import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

df = pd.read_csv('Machine_Downtime.csv')

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Filter to include only data from January 1, 2022 and beyond
df = df[df['Date'] >= '2021-12-20']
df = df[df['Date'] <= '2022-07-02']


# List of unique machines
machines = df['Machine_ID'].unique()

# Add a selectbox to choose a machine
machine = st.selectbox("Select Machine", machines)

# Filter data for the selected machine
machine_data = df[df['Machine_ID'] == machine]

# Sort the data by Date to ensure chronological order
machine_data = machine_data.sort_values(by='Date')

# 1. Downtime trend (line chart)
st.subheader(f"Downtime Trend for {machine}")

# Group by Date and calculate downtime events
downtime_trends = machine_data.groupby('Date')['Downtime'].value_counts().unstack(fill_value=0)
downtime_trends.columns = ['No Downtime', 'Downtime']  # Rename columns for clarity

# Create line chart
fig = px.line(downtime_trends, 
              x=downtime_trends.index, 
              y='Downtime', 
              title=f'Downtime Trend for {machine}', 
              labels={'Date': 'Date', 'Downtime': 'Downtime Events'})

# Display the plot
st.plotly_chart(fig)
