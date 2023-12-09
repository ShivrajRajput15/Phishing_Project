# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.externals import joblib  # For model persistence

# Load the PhishTank dataset
# You can download the dataset from https://www.phishtank.com/developer_info.php
# Assume the dataset is stored in a CSV file with columns 'url' and 'phishing'
dataset_path = 'path/to/phishtank_dataset.csv'
df = pd.read_csv(dataset_path)

# Data preprocessing
X = df['url']  # Features (URLs)
y = df['phishing']  # Target variable (0 for non-phishing, 1 for phishing)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert URLs to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Decision Tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model for future use
joblib.dump(classifier, 'phishing_detection_model.joblib')
