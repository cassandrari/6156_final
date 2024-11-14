import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

df = pd.read_csv('Machine_Downtime.csv')

machines = df['Machine_ID'].unique()

for machine in machines:
    # Filter data for the current machine
    machine_data = df[df['Machine_ID'] == machine]

    # Create columns for side-by-side graphs
    col1, col2, col3 = st.columns(3)

    # 1. Trend analysis graph (line chart for downtime trend)
    with col1:
        st.subheader(f"Downtime Trend for {machine}")
        downtime_trends = machine_data.groupby('Date')['Downtime'].value_counts().unstack(fill_value=0)
        downtime_trends.columns = ['No Downtime', 'Downtime']  # Rename columns for clarity
        fig = px.line(downtime_trends, 
                      x=downtime_trends.index, 
                      y='Downtime', 
                      title=f'Downtime Trend for {machine}', 
                      labels={'Date': 'Date', 'Downtime': 'Downtime Events'})
        st.plotly_chart(fig)
