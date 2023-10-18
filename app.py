from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from src.predicting_publications.pipeline.prediction import PredictionPipeline
from src.predicting_publications import logger

app = Flask(__name__)  # Initialize Flask

@app.route('/', methods=['GET'])
def home_page():
    """
    Render the home page of the application.

    This function handles the GET request to the root URL ('/') and renders 
    the index.html template which is the home page of the application.

    Returns:
    --------
    str
        Rendered HTML template.
    """
    return render_template("index.html")


@app.route('/train', methods=['GET'])
def train():
    """
    Route to initiate the training pipeline.
    
    This route triggers the main training script when accessed. It's designed 
    to be a simple way to start training the model remotely via a web request.
    
    Returns:
        str: A message indicating the training status.
    """
    # Execute the main training script
    # Note: Directly using os.system can be risky (especially if parameters 
    # come from user input). It's used here for simplicity. Consider more secure methods for production.
    os.system("python main.py")
    
    # Return a response indicating the completion of training
    return "Training Completed"


@app.route('/predict', methods=['POST', 'GET'])
def index():
    """
    Route to make predictions using the trained model.
    
    This route collects input features from the user via a web form,
    uses them to predict the target variable using a pre-trained model,
    and then displays the prediction result.

    Returns:
        str: Rendered HTML page displaying the prediction result or the input form.
    """
    if request.method == 'POST':
        try:
            # Extracting user input from the form
            lon = float(request.form['lon'])
            lat = float(request.form['lat'])
            hour = int(request.form['hour'])
            day = int(request.form['day'])
            dayofweek = int(request.form['dayofweek'])
            month = int(request.form['month'])
            likescount = float(request.form['likescount'])
            commentscount = float(request.form['commentscount'])
            symbols_cnt = float(request.form['symbols_cnt'])
            words_cnt = float(request.form['words_cnt'])
            hashtags_cnt = float(request.form['hashtags_cnt'])
            mentions_cnt = float(request.form['mentions_cnt'])
            links_cnt = float(request.form['links_cnt'])
            emoji_cnt = float(request.form['emoji_cnt'])
            
            # Organizing the data into a format suitable for prediction
            data = {
                'lon': [lon],
                'lat': [lat],
                'hour': [hour],
                'day': [day],
                'dayofweek': [dayofweek],
                'month': [month],
                'likescount': [likescount],
                'commentscount': [commentscount],
                'symbols_cnt': [symbols_cnt],
                'words_cnt': [words_cnt],
                'hashtags_cnt': [hashtags_cnt],
                'mentions_cnt': [mentions_cnt],
                'links_cnt': [links_cnt],
                'emoji_cnt': [emoji_cnt]
            }
            data_df = pd.DataFrame(data)
            
            # Making the prediction
            pipeline = PredictionPipeline()
            prediction = pipeline.predict(data_df)
            
            # Render and return the results page
            return render_template('results.html', prediction=str(prediction))
            
        except Exception as e:
            # Log the exception for debugging
            logger.error(f"Error occurred during prediction: {e}")

            # Re-raise the exception to be handled by Flask's default error handling
            raise e

    # If the request method is 'GET', just render the input form
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8383, debug=True)
