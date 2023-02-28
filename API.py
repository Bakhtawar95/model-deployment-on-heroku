import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
     

    features = [float(x) for x in request.form.values()]
    X = [np.array(features)] 
    prediction=model.predict(X)
    output=prediction[0]
    return render_template('index.html', prediction_text='The predicted stress level of the human is {}'.format(output))



#if __name__ == "__main__":
    #app.run(debug=True)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port="8080")