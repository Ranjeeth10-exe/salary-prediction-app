import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import joblib
import os

# Create dummy Salary_Data.csv if it doesn't exist for testing purposes
if not os.path.exists('Salary_Data.csv'):
    np.random.seed(42)
    exp = np.linspace(1, 15, 30)
    # Polynomial relationship: Salary = 30000 + 15000*exp + 2000*exp^2 + noise
    salary = 30000 + 15000 * exp + 2000 * (exp ** 2) + np.random.normal(0, 10000, 30)
    pd.DataFrame({'YearsExperience': exp, 'Salary': salary}).to_csv('Salary_Data.csv', index=False)
    print("Created dummy Salary_Data.csv")

# Load Dataset
dataset = pd.csv_read('Salary_Data.csv')
X = dataset[['YearsExperience']].values
y = dataset['Salary'].values

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Polynomial Regression Model (Degree 2)
# Using a Pipeline ensures the input is automatically transformed when predicting
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Train Model
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, 'model.pkl')
print("Model trained and saved as model.pkl")