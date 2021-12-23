from flask import Flask, render_template, request
from datetime import date
#import requests
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('rf_regressor_model.pkl', 'rb'))
encoder = pickle.load(open('model_encoder.pkl', 'rb'))

@app.route('/', methods = ['GET'])
def Home():
    return render_template('index.html') 


@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':

        today = date.today()
        manufacturer = encoder[0].transform([request.form['Manufacturer']])[0]
        condition = encoder[1].transform([request.form['Condition']])[0]
        cylinders = encoder[2].transform([request.form['Cylinders']])[0]
        fuel_type = encoder[3].transform([request.form['Fuel Type']])[0]
        distance = float(request.form['Distance'])
        status = encoder[4].transform([request.form['Status']])[0]
        transmission = encoder[5].transform([request.form['Transmission']])[0]
        drive = encoder[6].transform([request.form['Drive']])[0]
        car_type = encoder[7].transform([request.form['Car Type']])[0]
        state = encoder[8].transform([request.form['State']])[0]
        year_ = today.year - int(request.form['Year']) 

        prediction = model.predict([[manufacturer, condition, cylinders, fuel_type, distance, status, transmission, drive, car_type, state, year_]])
        output = int(prediction[0])
        return render_template('index.html', prediction_text = 'The best price estimated is {} $'.format(str(output)))

    else:
        return render_template('index.html')

 
if __name__=='__main__':
    app.run(debug=True)   