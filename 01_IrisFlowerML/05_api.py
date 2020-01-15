from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json

app = Flask(__name__)

@app.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    prediction = np.array2string(model.predict(data))
    return jsonify(prediction)

if __name__ == '__main__':
    modelfile = 'model.model'
    model = p.load(open(modelfile, 'rb'))
    app.run(debug=False)
    
# example request with json body "[[7.2,3.6,6.1,2.5]]" gives you the flower type