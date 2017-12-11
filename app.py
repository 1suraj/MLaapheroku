# -*- coding: utf-8 -*-


import numpy as np
from flask import Flask, abort, jsonify, request
#from flask_cors import cross_origin
#from sklearn.externals import joblib
#from json import dumps
import _pickle as pickle 

app = Flask(__name__)

my_RFmodel= pickle.load(open('RFmodel.pkl','rb'))
SC= pickle.load(open('SC_X.pkl','rb'))

@app.route('/api', methods=['POST'])
def make_predict():
    
    # This Cn be requested through 
    '''
    import requests, json
    url = 'http://localhost:5000/api1'
    data = json.dumps({'Age':'42','Salary':'50000'})
    r = requests.post(url, data)

    print(r.text)
    '''
    
    data = request.get_json(force=True)
    Age = data['Age']
    Salary = data['Salary']
    predict_request = [[Age,Salary]]
    predict_request=np.array(predict_request)
    predict_request=SC.transform(predict_request)
    predicted_result = my_RFmodel.predict(predict_request)
    output=[predicted_result][0]
    if output==0:
        prediction='Not purchases'
    else: prediction='purchase'
   
    
    #return test
    return jsonify(prediction)
    #return jsonify(results =a)

@app.route('/')
def hello_world():
    return 'Deployment Successful1!'
'''
@app.route('/api', methods=['POST'])
def hello_world1():
    data = request.get_json(force=True)
    data = data['no']
    return data

'''    
if __name__ == '__main__':
    app.run()
    
