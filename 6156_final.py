pip install plotly
import streamlit as st
import pandas as pd
import plotly.express as px


#dataset
machine_data = pd.read_csv('Machine_Downtime.csv')
machine_data['Date'] = pd.to_datetime(machine_data['Date'])


#trend analysis over time 
downtime_trends = machine_data.groupby('Date')['Downtime'].value_counts().unstack(fill_value=0)
downtime_trends.columns = ['No Downtime', 'Downtime']  # Rename columns for clarity

fig = px.line(downtime_trends, 
              x=downtime_trends.index, 
              y='Downtime', 
              title=f'Downtime Trend for {machine}',
              labels={'Date': 'Date', 'Downtime': 'Number of Downtime Events'})

st.plotly_chart(fig)
