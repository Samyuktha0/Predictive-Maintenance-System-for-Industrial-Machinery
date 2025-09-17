import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load processed data
data = pd.read_csv('data/processed_data.csv')

# Define features (X) and target (y)
features = [col for col in data.columns if 'sensor' in col]
X = data[features]
y = data['RUL']

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'models/predictive_model.joblib')

print("Model trained and saved.")
