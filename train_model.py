import pandas as pd
import joblib

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Load dataset
dataset = pd.read_csv('Salary_Data.csv')

# Input and Output
X = dataset[['YearsExperience']]
y = dataset['Salary']

# Polynomial Regression Model
model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')

print("Model trained successfully")