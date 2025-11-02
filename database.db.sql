# model_training.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# ======================
# STEP 1: Load FER2013 Dataset
# ======================
print("Loading FER2013 dataset...")
data = pd.read_csv("https://raw.githubusercontent.com/muxspace/facial_expressions/master/fer2013/fer2013.csv")

# Extract image data and labels
pixels = data['pixels'].tolist()
emotion_labels = data['emotion'].values

# Convert pixel strings to arrays
X = np.array([np.fromstring(p, dtype=int, sep=' ') for p in pixels])
X = X.reshape(-1, 48, 48, 1).astype('float32') / 255.0  # Normalize

y = to_categorical(emotion_labels, num_classes=7)  # FER2013 has 7 emotions

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dataset loaded successfully.")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ======================
# STEP 2: Data Augmentation
# ======================
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# ======================
# STEP 3: Build CNN Model
# ======================
print("Building CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ======================
# STEP 4: Train Model
# ======================
print("Training model... (this may take 20â40 minutes depending on your system)")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=25,
    validation_data=(X_test, y_test)
)

# ======================
# STEP 5: Save Model
# ======================
model.save("face_emotionModel.h5) 
print("â
 Model training complete. Saved as face_emotionModel.h5")
