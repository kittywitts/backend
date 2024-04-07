from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='http://localhost:3000')

@app.route('/', methods=['POST'])
def predict():
    try:
        with open('model.pkl', 'rb') as file:
            feature_extraction, model = pickle.load(file)
        email = request.json['email']  # Adjusted to 'email'
        input_data_features = feature_extraction.transform([email])  # Adjusted to 'email'
        prediction = model.predict(input_data_features)
        result = 1 if prediction[0] == 1 else 0
        return jsonify(result)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': 'An error occurred during prediction'})

if __name__ == '__main__':
    app.run(debug=True)
