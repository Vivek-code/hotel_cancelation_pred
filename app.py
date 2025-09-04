# # app.py
# import numpy as np
# import pandas as pd
# from flask import Flask, request, jsonify, render_template
# import joblib
# import sqlite3
# import os

# # Initialize the flask app
# app = Flask(__name__)

# # --- Load the trained model and column data ---
# # Make sure these files are in the root directory
# try:
#     model = joblib.load('model.joblib')
#     model_columns = joblib.load('model_columns.pkl')
#     print("Model and columns loaded successfully.")
# except FileNotFoundError as e:
#     print(f"Error loading model files: {e}")
#     print("Please ensure 'model.joblib' and 'model_columns.pkl' are in the root directory.")
#     model = None
#     model_columns = None

# # --- Database Connection ---
# DATABASE_FOLDER = 'database'
# DATABASE_PATH = os.path.join(DATABASE_FOLDER, 'trends.db')

# def get_db_connection():
#     """Establishes a connection to the SQLite database."""
#     conn = sqlite3.connect(DATABASE_PATH)
#     conn.row_factory = sqlite3.Row
#     return conn

# # --- Define web routes ---

# @app.route('/')
# def home():
#     """Renders the main prediction page."""
#     return render_template('index.html')

# @app.route('/visualizations')
# def visualizations():
#     """Renders the EDA visualizations page."""
#     return render_template('visualizations.html')

# @app.route('/trends')
# def trends():
#     """Fetches trend data from the DB and renders the trends page."""
#     conn = get_db_connection()
    
#     # Fetch monthly and weekly trends
#     monthly_data = conn.execute('SELECT arrival_date_month, cancellation_count FROM monthly_trends').fetchall()
#     weekly_data = conn.execute('SELECT day_of_week, cancellation_count FROM weekly_trends').fetchall()
    
#     conn.close()
    
#     # --- FIX: Sort data chronologically in Python ---
#     # The database sorts alphabetically, so we re-order it here.
#     month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
#     day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

#     # Convert fetched data to a dictionary for easier lookup
#     monthly_dict = {row['arrival_date_month']: row['cancellation_count'] for row in monthly_data}
#     weekly_dict = {row['day_of_week']: row['cancellation_count'] for row in weekly_data}

#     # Prepare data for Chart.js in the correct order
#     month_labels = month_order
#     month_values = [monthly_dict.get(month, 0) for month in month_order]
    
#     day_labels = day_order
#     day_values = [weekly_dict.get(day, 0) for day in day_order]
    
#     return render_template('trends.html', 
#                            month_labels=month_labels, 
#                            month_values=month_values,
#                            day_labels=day_labels,
#                            day_values=day_values)


# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handles the prediction request from the form."""
#     if model is None or model_columns is None:
#         return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

#     try:
#         # Get data from POST request
#         data = request.get_json(force=True)
        
#         # Create a DataFrame from the input data
#         query_df = pd.DataFrame([data])
        
#         # --- Preprocess the input data ---
#         # Ensure all required columns are present and in the correct order
#         query_df = query_df.reindex(columns=model_columns, fill_value=0)
        
#         # Apply the same log transformations as in training
#         log_cols = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month', 'agent', 'company', 'adr']
#         for col in log_cols:
#             query_df[col] = np.log1p(query_df[col].astype(float))
            
#         # Make prediction
#         prediction = model.predict(query_df)
        
#         # Get prediction probability
#         prediction_proba = model.predict_proba(query_df)
        
#         # Format the output
#         if prediction[0] == 1:
#             output = "Booking will be Cancelled"
#             probability = f"{prediction_proba[0][1]*100:.2f}%"
#         else:
#             output = "Booking will not be Cancelled"
#             probability = f"{prediction_proba[0][0]*100:.2f}%"
            
#         return jsonify({'prediction_text': output, 'probability': probability})

#     except Exception as e:
#         print(f"Prediction Error: {e}")
#         return jsonify({'error': str(e)}), 400


# if __name__ == "__main__":
#     app.run(debug=True)
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import joblib
import os

# Initialize the flask app
app = Flask(__name__)
app.secret_key = 'super_secret_key' # Needed for flashing messages

# --- Load the trained model and column data ---
try:
    model = joblib.load('model.joblib')
    model_columns = joblib.load('model_columns.pkl')
    print("Model and columns loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    model = None
    model_columns = None

# --- Preprocessing function for batch data ---
def preprocess_batch_data(df):
    """Applies the same preprocessing steps from training to the uploaded dataframe."""
    df_processed = df.copy()
    df_processed['company'] = df_processed['company'].fillna(0)
    df_processed['agent'] = df_processed['agent'].fillna(0)
    df_processed['country'] = df_processed['country'].fillna('Undefined')
    df_processed['children'] = df_processed['children'].fillna(0)
    df_processed['hotel'] = df_processed['hotel'].map({'Resort Hotel': 0, 'City Hotel': 1})
    df_processed['arrival_date_month'] = df_processed['arrival_date_month'].map({'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12})
    df_processed['meal'] = df_processed['meal'].map({'BB': 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4})
    df_processed['market_segment'] = df_processed['market_segment'].map({'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3, 'Complementary': 4, 'Groups': 5, 'Undefined': 6, 'Aviation': 7})
    df_processed['distribution_channel'] = df_processed['distribution_channel'].map({'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'Undefined': 3, 'GDS': 4})
    df_processed['reserved_room_type'] = df_processed['reserved_room_type'].map({'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6, 'L': 7, 'B': 8, 'P': 9})
    df_processed['deposit_type'] = df_processed['deposit_type'].map({'No Deposit': 0, 'Refundable': 1, 'Non Refund': 2})
    df_processed['customer_type'] = df_processed['customer_type'].map({'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3})
    df_processed = df_processed.fillna(0)
    log_cols = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month', 'agent', 'company', 'adr']
    for col in log_cols:
        df_processed[col] = np.log1p(df_processed[col].astype(float))
    df_processed = df_processed.reindex(columns=model_columns, fill_value=0)
    return df_processed

# --- Define web routes ---

@app.route('/')
def home():
    """Renders the main prediction page."""
    return render_template('index.html')

@app.route('/visualizations')
def visualizations():
    """Renders the EDA visualizations page."""
    return render_template('visualizations.html')

@app.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    """Handles file upload and batch prediction."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                original_df = df.copy()
                processed_df = preprocess_batch_data(df)
                predictions = model.predict(processed_df)
                probabilities = model.predict_proba(processed_df)[:, 1]
                original_df['Predicted Cancellation'] = ['Yes' if p == 1 else 'No' for p in predictions]
                original_df['Cancellation Probability'] = [f"{p*100:.2f}%" for p in probabilities]
                total_bookings = len(original_df)
                total_cancellations = np.sum(predictions)
                
                # Use a month map that includes all months to avoid errors
                month_map_to_name = {
                    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 
                    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
                }
                month_map_to_num = {v: k for k, v in month_map_to_name.items()}

                # Convert numeric month from preprocessing back to name for display
                original_df['arrival_date_month_name'] = original_df['arrival_date_month'].map(month_map_to_name.get)
                monthly_cancellations = original_df[original_df['Predicted Cancellation'] == 'Yes']['arrival_date_month_name'].value_counts().to_dict()
                
                recommendation = "Low cancellation risk detected. Standard operational procedures are likely sufficient."
                if total_bookings > 0 and total_cancellations / total_bookings > 0.5:
                    recommendation = "High cancellation risk! Consider implementing a strategic overbooking policy for high-risk months and focus on guest communication to confirm bookings."
                elif total_bookings > 0 and total_cancellations / total_bookings > 0.2:
                    recommendation = "Moderate cancellation risk. It is advisable to send confirmation emails to guests with high cancellation probability and review booking policies for flexibility."

                results_html = original_df[['hotel', 'arrival_date_month_name', 'lead_time', 'adr', 'Predicted Cancellation', 'Cancellation Probability']].to_html(classes='table table-striped table-hover', index=False)
                
                return render_template('batch.html', 
                                       results_html=results_html, 
                                       total_bookings=total_bookings,
                                       total_cancellations=total_cancellations,
                                       monthly_cancellations=monthly_cancellations,
                                       recommendation=recommendation)
            except Exception as e:
                flash(f'An error occurred: {e}', 'danger')
                return redirect(request.url)

    # --- THIS IS THE FIX ---
    # For a GET request, render the template with 'results_html' set to None.
    # This tells the template not to try and display the results section.
    return render_template('batch.html', results_html=None)


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the single prediction request from the form."""
    if model is None or model_columns is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500
    try:
        data = request.get_json(force=True)
        query_df = pd.DataFrame([data])
        query_df = query_df.reindex(columns=model_columns, fill_value=0)
        log_cols = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month', 'agent', 'company', 'adr']
        for col in log_cols:
            query_df[col] = np.log1p(query_df[col].astype(float))
        prediction = model.predict(query_df)
        prediction_proba = model.predict_proba(query_df)
        if prediction[0] == 1:
            output = "Booking will be Cancelled"
            probability = f"{prediction_proba[0][1]*100:.2f}%"
        else:
            output = "Booking will not be Cancelled"
            probability = f"{prediction_proba[0][0]*100:.2f}%"
        return jsonify({'prediction_text': output, 'probability': probability})
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)



