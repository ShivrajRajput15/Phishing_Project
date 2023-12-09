from flask import Flask, request, jsonify
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
classifier = joblib.load('phishing_detection_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')  # Save the vectorizer during training

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    url = data['url']

    # Transform the input URL using the loaded TF-IDF vectorizer
    url_tfidf = vectorizer.transform([url])

    # Make prediction using the loaded model
    prediction = classifier.predict(url_tfidf)[0]

    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(port=5000)
