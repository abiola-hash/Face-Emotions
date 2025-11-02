# FACE_DETECTION — Facial Emotion Detector Web App

## Overview
A simple Flask web app that accepts a face photo, runs a trained Keras model (face_emotionModel.h5) to predict an emotion, and stores the user's details + filename + prediction in an SQLite database.

## Files
- app.py — main Flask app
- model_training.py — example training script (replace placeholder data with FER2013 or your prepared dataset)
- templates/ — index.html and result.html
- requirements.txt — python dependencies
- database.db — created by the app on first run
- face_emotionModel.h5 — put your trained model here

## Setup (local)
1. Create a Python virtual environment (recommended).
2. Install dependencies: pip install -r requirements.txt
3. Place your trained face_emotionModel.h5 in the project root.
4. Run: python app.py
5. Open http://127.0.0.1:5000

## Notes
- The model expects 48x48 grayscale input (FER2013 style). Adjust preprocessing if your model uses different shape or RGB.
- The database will be created automatically. Uploaded images are kept in uploads/.

## Deploy to Render
1. Add a Procfile with: web: gunicorn app:app
2. Push to GitHub and connect the repo to Render.
3. Set environment variables if needed (FLASK_SECRET).
4. Ensure face_emotionModel.h5 is included or use Render's file upload steps or S3 to host the model and adjust MODEL_PATH.
