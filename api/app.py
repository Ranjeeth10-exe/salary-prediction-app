import os
from flask import Flask, request, render_template
import joblib
import numpy as np

# Absolute path configurations for Vercel deployment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

def format_inr(number):
    """Formats a number into Indian Rupee string format."""
    num_str = str(int(number))
    if len(num_str) <= 3:
        return f"₹ {num_str}"
    
    result = num_str[-3:]
    num_str = num_str[:-3]
    
    while len(num_str) > 2:
        result = num_str[-2:] + "," + result
        num_str = num_str[:-2]
        
    if num_str:
        result = num_str + "," + result
        
    return f"₹ {result}"

@app.route('/', methods=['GET', 'POST'])
def index():
    annual_salary = None
    monthly_salary = None
    error = None

    if request.method == 'POST':
        if not model:
            error = "Model not found. Please train the model first."
            return render_template('index.html', error=error)

        try:
            years_exp = float(request.form['experience'])
            
            if years_exp < 0:
                error = "Experience cannot be negative."
            else:
                # Predict (model pipeline handles the poly transform automatically)
                prediction = model.predict(np.array([[years_exp]]))[0]
                
                # Prevent negative salary predictions for zero/low experience
                prediction = max(0, prediction)
                
                monthly_pred = prediction / 12

                annual_salary = format_inr(prediction)
                monthly_salary = format_inr(monthly_pred)
                
        except ValueError:
            error = "Please enter a valid number."

    return render_template(
        'index.html', 
        annual_salary=annual_salary, 
        monthly_salary=monthly_salary, 
        error=error
    )

# Required for local testing
if __name__ == '__main__':
    app.run(debug=True)