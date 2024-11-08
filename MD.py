import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Machine_Downtime.csv')

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows where 'Date' is NaT (if any)
df.dropna(subset=['Date'], inplace=True)

# Visualize the distribution of numerical features
numerical_features = df.select_dtypes(include=[np.number]).columns

plt.figure(figsize=(14, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(4, 4, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Correlation Heatmap: Visualize correlation between numerical features
plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

from sklearn.preprocessing import LabelEncoder

# Check unique values in 'Downtime' column
print(df['Downtime'].unique())

# If 'Downtime' contains categorical labels like 'No_Machine_Failure', 'Machine_Failure'
# Convert 'Downtime' to numeric values (e.g., 0 for 'No_Machine_Failure' and 1 for 'Machine_Failure')
label_encoder = LabelEncoder()
df['Downtime'] = label_encoder.fit_transform(df['Downtime'])

# Verify that the conversion was successful
print(df['Downtime'].unique())

# Now you can proceed with the model training
# Define features (X) and target variable (y)
X = df[['Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)', 'Air_System_Pressure(bar)', 
        'Coolant_Temperature', 'Hydraulic_Oil_Temperature(?C)', 'Spindle_Bearing_Temperature(?C)',
        'Spindle_Vibration(?m)', 'Tool_Vibration(?m)', 'Spindle_Speed(RPM)', 'Voltage(volts)', 
        'Torque(Nm)', 'Cutting(kN)', 'Rolling_Hydraulic_Temperature']]

# Target variable: Downtime (now numeric)
y = df['Downtime']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared: {r2}')

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Convert continuous predictions to class labels (0 or 1)
y_pred_class = (y_pred > 0.5).astype(int)

# Calculate Precision, Recall, F1-Score
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)

# Print the metrics and confusion matrix
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print('Confusion Matrix:')
print(cm)
