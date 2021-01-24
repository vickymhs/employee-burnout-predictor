from flask import Flask, render_template, request, redirect, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
from tensorflow.keras.models import Sequential, save_model, load_model

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

app = Flask(__name__)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

graph = tf.get_default_graph()
prediction = None
# new_model = load_model("whole_model")
print("Loaded model from disk")

#route() decorators
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #Getting form data from the web form
    #print(request)
    name = request.form['name']
    gender = int(request.form['gender'])
    wfh_setup = int(request.form['wfh_setup'])
    company_type = int(request.form['company_type'])
    designation = float(request.form['designation'])
    resource_allocation = float(request.form['resource_allocation'])
    mental_fatigue_score = float(request.form['mental_fatigue_score'])

    print("gender type ", type(gender))
    print("wfh type ", type(wfh_setup))
    print("company type ", type(company_type))
    print("designation type ", type(designation))
    print("resource_allocation type", type(resource_allocation))
    print("mental_fatigue_score type", type(mental_fatigue_score))

    final_features = np.array([[gender, wfh_setup, company_type, designation, resource_allocation, mental_fatigue_score]])
    # df = pd.DataFrame([lst])
    # df.columns =['Gender_Encoded','WFH_Setup_Encoded','Company_Type_Encoded','Designation','Resource Allocation','Mental Fatigue Score']
    print(final_features)
    print(final_features.shape)
    
    # loaded_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    with graph.as_default():
    	prediction = loaded_model.predict(final_features)
    Burnout = prediction[0][0] * 100
    print(Burnout)
    output = round(Burnout, 2)
    return render_template('index.html', result = "Your Burnout rate is {}".format(output))


if __name__ == '__main__':
    app.run(debug=True)