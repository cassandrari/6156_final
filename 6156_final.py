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

    # 2. Scatter plot showing the relationship between temperature and pressure
    with col2:
        st.subheader(f"Temperature vs Pressure for {machine}")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(machine_data['Hydraulic_Pressure(bar)'], machine_data['Torque(Nm)'], color='blue')
        ax.set_xlabel('Hydraulic Pressure')
        ax.set_ylabel('Torque (Nm)')
        ax.set_title(f'Hydraulic Pressure vs Torque for {machine}')
        st.pyplot(fig)

    # 3. Bar chart showing the count of downtime events (Y vs N)
    with col3:
        st.subheader(f"Downtime Distribution for {machine}")
        downtime_counts = machine_data['Downtime'].value_counts()
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.bar(downtime_counts.index, downtime_counts.values, color=['red', 'green'])
        ax.set_xlabel('Downtime Status')
        ax.set_ylabel('Count')
        ax.set_title(f"Downtime Status Distribution for {machine}")
        st.pyplot(fig)
