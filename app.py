from flask import Flask, render_template, request
import pandas as pd
import os
import pickle
import numpy as np

app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model_path = "model.pkl"
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Function to process uploaded CSV and extract date-related features
def process_data(df):
    df = pd.read_csv('BTC-USD.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Day of Week'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
    df['Week of Year'] = df['Date'].dt.isocalendar().week
    df['Day of Year'] = df['Date'].dt.dayofyear
    df['Month_sin'] = np.sin((df['Month']-1)*(2.*np.pi/12))
    df['Month_cos'] = np.cos((df['Month']-1)*(2.*np.pi/12))
    df['Day of Year_sin'] = np.sin(2 * np.pi * df['Day of Year'] / 365)
    df['Day of Year_cos'] = np.cos(2 * np.pi * df['Day of Year'] / 365)
    return df.head().to_html()

# Flask route for uploading files and predicting values
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    table = None

    if request.method == 'POST':
        # Handling File Upload
        

        # Handling Manual Input for Prediction
        if all(key in request.form for key in ['Date', 'Open', 'High', 'Low', 'Adj_Close', 'Volume']):
            try:
                # Extracting Date and Splitting into Components
                date_time_str = request.form['Date']  
                date_time = pd.to_datetime(date_time_str, errors='coerce')

                if pd.isnull(date_time):
                    raise ValueError("Invalid date format. Please provide a valid date (YYYY-MM-DD).")

        
                # Convert input to DataFrame
                input_data = pd.DataFrame({
                    'Open': [float(request.form['Open'])],
                    'High': [float(request.form['High'])],
                    'Low': [float(request.form['Low'])],
                    'Adj Close': [float(request.form['Adj_Close'])],
                    'Volume': [float(request.form['Volume'])],
                    'Year': date_time.year,
                    'Month': date_time.month,
                    'Day': date_time.day,
                    'Day of Week': date_time.dayofweek,  # Monday=0, Sunday=6
                    'Week of Year': date_time.isocalendar()[1],
                    'Day of Year': date_time.timetuple().tm_yday,
                    'Month_sin': np.sin((date_time.month-1)*(2.*np.pi/12)),
                    'Month_cos': np.cos((date_time.month-1)*(2.*np.pi/12)),
                    'Day of year_sin': np.sin(2 * np.pi * date_time.timetuple().tm_yday / 365),
                    'Day of year_cos': np.cos(2 * np.pi * date_time.timetuple().tm_yday / 365)
                })

                # Ensure model is loaded
                if model:
                    prediction = model.predict(input_data)[0]  # Make prediction
                else:
                    prediction = "Model not loaded!"
            except Exception as e:
                prediction = f"Error: {e}"

    return render_template('index.html', table=table, prediction=prediction)


