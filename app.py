# app.py
from flask import Flask, render_template, request
import sqlite3
import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from datetime import datetime

# --- Configuration ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
DB_FILE = 'database.db'

# Create folders if missing
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = 'face_emotionModel.h5'
model = load_model(MODEL_PATH)

# Emotion labels (same order as training)
EMOTIONS = ["happy", "sad", "surprised", "angry", "neutral"]


# --- Database setup ---
def init_db():
    """Create database if not already present"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    email TEXT,
                    department TEXT,
                    image_filename TEXT,
                    prediction TEXT,
                    date_uploaded TEXT
                )''')
    conn.commit()
    conn.close()


def save_to_db(name, email, department, image_filename, prediction):
    """Save user info + image + prediction to DB"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    date_uploaded = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO users (name, email, department, image_filename, prediction, date_uploaded) VALUES (?, ?, ?, ?, ?, ?)",
              (name, email, department, image_filename, prediction, date_uploaded))
    conn.commit()
    conn.close()


# --- Image preprocessing ---
def preprocess_image(filepath):
    """Read image and prepare it for the model"""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))  # same size as training
    img = img / 255.0
    img = img.reshape(1, 48, 48, 1)
    return img


# --- Flask routes ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', result="No image uploaded!")

    # Get form data
    name = request.form['name']
    email = request.form['email']
    department = request.form['department']
    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', result="No file selected!")

    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess and predict
    img_array = preprocess_image(filepath)
    preds = model.predict(img_array)
    emotion_index = np.argmax(preds)
    predicted_emotion = EMOTIONS[emotion_index]

    # Create readable response
    messages = {
        "happy": "You look happy! Keep smiling 😊",
        "sad": "You are frowning. Why are you sad?",
        "surprised": "Wow! You look surprised 😲",
        "angry": "You seem angry 😠. Take it easy!",
        "neutral": "You look calm and neutral."
    }
    result_text = messages.get(predicted_emotion, predicted_emotion)

    # Save info to database
    save_to_db(name, email, department, file.filename, result_text)

    # Display result
    return render_template('index.html', result=result_text, filename=file.filename)


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
