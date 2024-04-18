from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from pandas import DataFrame

app = Flask(__name__)
CORS(app)

loaded_model = joblib.load('model.pkl') 

@app.route('/predict', methods=['POST'])
def receive_data():
    data = request.get_json()  
    data = {key: [value] for key, value in data.items()}
    df = DataFrame(data) 
    prediction = loaded_model.predict(df) 
    return jsonify({'prediction':prediction.tolist()})  

if __name__ == '__main__':
    app.run(debug=True)
