import numpy as np
from flask import Flask,request,render_template
import joblib
app = Flask(__name__)
model = joblib.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    print(request.form.values())
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    print(prediction)
    output =  round(prediction[0][0],2)


    return render_template('index.html',prediction_text ='Estimated Price Of The House must be ${}'.format(output))
if __name__ =='__main__':
    app.run(debug=True)