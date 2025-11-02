from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
DATABASE = 'database.db'
MODEL_PATH = 'face_emotionModel.h5'  # place the trained model here

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.environ.get('FLASK_SECRET', 'dev_secret_change_me')

# Load the trained model (lazy load when first request)
model = None
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fullname TEXT,
        email TEXT,
        matric TEXT,
        filename TEXT,
        predicted_emotion TEXT,
        timestamp TEXT
    )
    ''')
    conn.commit()
    conn.close()


def preprocess_image(image_bytes, target_size=(48,48)):
    # Convert bytes to grayscale 48x48 as used in FER2013 model
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize(target_size)
    arr = np.asarray(image, dtype=np.float32)
    arr = arr / 255.0
    arr = arr.reshape(1, target_size[0], target_size[1], 1)
    return arr


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    global model
    fullname = request.form.get('fullname', '').strip()
    email = request.form.get('email', '').strip()
    matric = request.form.get('matric', '').strip()

    if 'photo' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['photo']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_" + file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file_bytes = file.read()
        with open(path, 'wb') as f:
            f.write(file_bytes)

        # Load model if not already loaded
        if model is None:
            if not os.path.exists(MODEL_PATH):
                flash('Model file not found on server. Please upload face_emotionModel.h5 to the project root.')
                return redirect(url_for('index'))
            model = load_model(MODEL_PATH)

        # Preprocess and predict
        x = preprocess_image(file_bytes)
        preds = model.predict(x)
        emotion_idx = int(np.argmax(preds, axis=1)[0])
        emotion = EMOTIONS[emotion_idx]

        # Save to database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('INSERT INTO users (fullname, email, matric, filename, predicted_emotion, timestamp) VALUES (?,?,?,?,?,?)',
                    (fullname, email, matric, filename, emotion, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

        return render_template('result.html', emotion=emotion, fullname=fullname, filename=filename)

    else:
        flash('File type not allowed. Please upload PNG/JPG/JPEG.')
        return redirect(url_for('index'))


if name == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
