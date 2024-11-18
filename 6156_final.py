import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# dataset
df = pd.read_csv('Machine_Downtime.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] >= '2021-12-20']
machines = df['Machine_ID'].unique()

# select box
machine = st.selectbox("Select Machine", ["Overview of All"] + list(machines))
machine_data = df[df['Machine_ID'] == machine] if machine != "Overview of All" else df

# Trend analysis
machine_data = machine_data.sort_values(by='Date')
st.markdown(f"<h3 style='text-align: center;'>Downtime Trends by Month</h3>", unsafe_allow_html=True)
downtime_trends = machine_data.groupby('Date')['Downtime'].sum()
fig = px.line(downtime_trends, 
              x=downtime_trends.index, 
              y=downtime_trends, 
              labels={'Date': 'Date', 'Downtime': 'Downtime Events'})
st.plotly_chart(fig)

# Proportion by month table
df['Month'] = df['Date'].dt.to_period('M')
monthly_downtime_all_machines = df.groupby(['Machine_ID', 'Month', 'Downtime']).size().unstack(fill_value=0)
monthly_downtime_all_machines.columns = ['No Machine Failure', 'Machine Failure']
monthly_downtime_all_machines['Total_Days'] = monthly_downtime_all_machines['No Machine Failure'] + monthly_downtime_all_machines['Machine Failure']
monthly_downtime_all_machines['Downtime_Percentage'] = (monthly_downtime_all_machines['Machine Failure'] / monthly_downtime_all_machines['Total_Days']) * 100
monthly_downtime_all_machines = monthly_downtime_all_machines.reset_index()
monthly_downtime_all_machines['Month'] = monthly_downtime_all_machines['Month'].astype(str)

min_month = monthly_downtime_all_machines['Month'].min()
max_month = monthly_downtime_all_machines['Month'].max()
monthly_downtime_all_machines = monthly_downtime_all_machines[(monthly_downtime_all_machines['Month'] > min_month) & (monthly_downtime_all_machines['Month'] < max_month)]

# If "Overview of All" is selected, show a downtime ranking for all machines per month
if machine == "Overview of All":
    ranking_df = monthly_downtime_all_machines.groupby(['Month', 'Machine_ID'])['Downtime_Percentage'].sum().reset_index()
    ranking_df['Rank'] = ranking_df.groupby('Month')['Downtime_Percentage'].rank(ascending=False, method='min')
    ranking_df = ranking_df.sort_values(by=['Month', 'Rank'])
    st.markdown(f"<h3 style='text-align: center;'>Machine Downtime Rankings by Month</h3>", unsafe_allow_html=True)
    st.table(ranking_df[['Month', 'Machine_ID', 'Downtime_Percentage', 'Rank']])
else:
    # Show downtime proportion for the selected machine
    selected_machine_data = monthly_downtime_all_machines[monthly_downtime_all_machines['Machine_ID'] == machine]
    downtime_proportion_table = selected_machine_data.pivot(index='Machine_ID', columns='Month', values='Downtime_Percentage')
    downtime_proportion_table = downtime_proportion_table.drop(columns='Machine_ID', errors='ignore')
    st.markdown(f"<h3 style='text-align: center;'>Downtime Proportion by Month</h3>", unsafe_allow_html=True)
    st.table(downtime_proportion_table)
