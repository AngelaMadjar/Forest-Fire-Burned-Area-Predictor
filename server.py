import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from scipy.special import inv_boxcox

# initializing a Flask app
app = Flask(__name__)

# loading the saved model
model = pickle.load(open('model.pkl', 'rb'))

# loading the saved scaler (the input data should follow the same preprocessing pipeline performed in the training phase)
scaler = pickle.load(open('scaler.pkl', 'rb'))

# since the model is trained over a transformed dataset (with a box cox transformation performed on the dependent variable in order to make it more balanced 
# and with values increased by 1 since the box cox transformation is undefined for values <= 0) the output should be reverted to its 'original' scope of values
def transform_output(output):
    inv = inv_boxcox(output, -0.5158436578881084) # -0.5158436578881084 is the optimal lambda value - the one which results in the best approximation of a normal distribution curve
    return inv-1                                  # that is automatically calculated by the box cox transformation

# exposing the api to http://localhost:5000/submit
@app.route('/submit', methods=['POST'])
def predict():
    data = {
        "month": int(request.form["month"]),
        "day": int(request.form['day']),
        "ffmc": float(request.form["ffmc"]),
        "dmc": float(request.form["dmc"]),
        "dc": float(request.form["dc"]),
        "isi": float(request.form["isi"]),
        "temperature": float(request.form["temperature"]),
        "rh": float(request.form["rh"]),
        "wind": float(request.form["wind"]),
        "rain": float(request.form["rain"])
    }

    input_data = np.array(list(data.values()))   

    scaled_input_data = scaler.transform(input_data.reshape(1,-1))
    prediction = model.predict(scaled_input_data)  # using the trained model to predict the burned area
    output = transform_output(prediction[0]) # transforming the prediction accordingly
    
    return jsonify("The predicted burned area covers " + str(round(output, 2)) + " ha")

@app.route('/')
def index():
    return render_template('index.html')

# setting the port
if __name__ == '__main__':
    app.run(port=5000, debug=True)
