from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return "Welcome to Fake News Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'Text not provided'}), 400
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    label = 'REAL' if prediction == 1 else 'FAKE'
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
# To run the app, use the command: python app.py
