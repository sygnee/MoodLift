import os
import numpy as np
import librosa
import tensorflow as tf
import cv2
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
import soundfile as sf
import bcrypt
import jwt
import datetime
from werkzeug.utils import secure_filename
import json



# Spotify API Credentials
SPOTIFY_CLIENT_ID = "d5eda56afcbf4ecbafef48fe3db8168c"
SPOTIFY_CLIENT_SECRET = "5694184341e04cd0817da569ad0dc3d8"


# Flask Setup
app = Flask(__name__)
CORS(app)

# Use environment variables for security
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_default_secret_key')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your_default_jwt_secret_key')


# Load the ML Model
try:
    model = tf.keras.models.load_model("D:/BE/BE project flask api/Speech-Emotion-Recogniton/emotion_recognition_model.keras")
    print(model.summary())  # Debugging: Check if the model is loaded
    # Print input shape
    print("Model Input Shape:", model.input_shape)
except Exception as e:
    print("❌ Error loading model:", str(e))
    model = None

# Emotion categories
emotions = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprise"]

# Emotion categories and uplifting playlist mapping
emotion_playlists = {
    "neutral": "Feel Good Indie Rock",
    "calm": "Peaceful Piano",  # ✅ Added Missing Emotion
    "happy": "Happy Hits!",
    "sad": "Cheer Up!",
    "angry": "Calm Vibes",
    "fear": "Confidence Boost",
    "disgust": "Feel-Good Classics",
    "surprise": "Good Vibes"
}


# Extract emotion labels (for model prediction)
emotions = list(emotion_playlists.keys())

# Dummy User Database (Replace with a real database in production)
users = {}

# Spotify API Credentials
SPOTIFY_CLIENT_ID = "d5eda56afcbf4ecbafef48fe3db8168c"
SPOTIFY_CLIENT_SECRET = "5694184341e04cd0817da569ad0dc3d8"

# =========================== Spotify API Authentication ===========================
def get_spotify_access_token():
    url = "https://accounts.spotify.com/api/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": "d5eda56afcbf4ecbafef48fe3db8168c",
        "client_secret": "5694184341e04cd0817da569ad0dc3d8",
    }
    response = requests.post(url, headers=headers, data=data)
    return response.json().get("access_token") if response.status_code == 200 else None

# =========================== Get Uplifting Playlist from Spotify ===========================
def get_spotify_playlist(emotion):
    token = get_spotify_access_token()
    if not token:
        return []

    headers = {"Authorization": f"Bearer {token}"}
    playlist_name = emotion_playlists.get(emotion, "Good Vibes")
    search_url = f"https://api.spotify.com/v1/search?q={playlist_name}&type=playlist&limit=5"

    response = requests.get(search_url, headers=headers)
    
    if response.status_code == 200:
        playlists = response.json().get("playlists", {}).get("items", [])
        return [{"name": p["name"], "url": p["external_urls"]["spotify"]} for p in playlists]
    else:
        print("❌ Error fetching playlists:", response.json())
        return []


# =========================== User Authentication ===========================
import json

USER_DB = "users.json"

def save_users():
    with open(USER_DB, "w") as f:
        json.dump(users, f)

def load_users():
    global users
    try:
        with open(USER_DB, "r") as f:
            users = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        users = {}

# Load users at startup
load_users()

import json

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get("username").lower()
    password = data.get("password")

    # Load users
    try:
        with open("users.json", "r") as f:
            users = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        users = {}

    if username in users:
        return jsonify({"error": "User already exists"}), 400

    # Store hashed password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    users[username] = hashed_password

    with open("users.json", "w") as f:
        json.dump(users, f)

    return jsonify({"message": "User registered successfully"}), 201


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username").lower()
    password = data.get("password")

    try:
        with open("users.json", "r") as f:
            users = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        users = {}

    if username not in users:
        return jsonify({"error": "User does not exist"}), 401

    stored_hashed_password = users[username].encode('utf-8')

    if not bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password):
        return jsonify({"error": "Invalid credentials"}), 401

    token = jwt.encode({"username": username, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)}, 
                        app.config['JWT_SECRET_KEY'], algorithm="HS256")

    return jsonify({"message": "Login successful", "token": token})


# =========================== Process Audio  ===========================
import librosa
import numpy as np
import io
import soundfile as sf

def process_audio(file_stream):
    try:
        print("🎤 Processing in-memory raw audio...")

        # ✅ Read the uploaded audio file into memory
        file_bytes = file_stream.read()
        file_buffer = io.BytesIO(file_bytes)

        # ✅ Load the raw audio file directly from memory
        y, sr = sf.read(file_buffer, dtype='float32')  
        print(f"🔹 Loaded Audio: {len(y)} samples at {sr} Hz")

        # ✅ Convert stereo to mono (if necessary)
        if len(y.shape) > 1:
            y = librosa.to_mono(y.T)  # Convert stereo to mono

        # ✅ Normalize audio (ensure values are in the same range)
        y = librosa.util.normalize(y)

        # ✅ Ensure the audio length is exactly 309 samples
        if len(y) < 309:
            y = np.pad(y, (0, 309 - len(y)), mode='constant')  # Pad if too short
        else:
            y = y[:309]  # Trim if too long

        print(f"📏 Final Processed Audio Shape for Model: {y.shape}")

        # ✅ **Fix: Reshape to match model input (batch_size=1, time_steps=309, features=1)**
        y = y.reshape(1, 309, 1)  # **This was missing!**

        print(f"📏 Reshaped Audio for Model: {y.shape}")  # Debugging

        return y

    except Exception as e:
        print(f"❌ Error processing audio: {str(e)}")
        return None



# =========================== Predict Emotion ===========================
# API Endpoint for Emotion Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("🎤 Received request for emotion prediction")

        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided!"}), 400

        audio_file = request.files["audio"]
        processed_audio = process_audio(audio_file)

        if processed_audio is None:
            return jsonify({"error": "Error processing audio"}), 500

        # ✅ Ensure processed_audio shape matches model input
        if processed_audio.shape != (1, 309, 1):
            print(f"❌ Invalid Input Shape! Expected (1, 309, 1) but got {processed_audio.shape}")
            return jsonify({"error": "Processed audio shape mismatch"}), 500

        # ✅ Predict emotion
        prediction = model.predict(processed_audio)

        # ✅ Debugging: Print model output shape BEFORE argmax
        print("📊 Model Raw Prediction Output:", prediction)
        print("📊 Model Prediction Shape:", prediction.shape)

        # ✅ Ensure model output has correct shape
        if prediction.shape[1] != len(emotions):
            return jsonify({
                "error": "Model output shape mismatch!",
                "expected_shape": (1, len(emotions)),
                "actual_shape": prediction.shape
            }), 500

        predicted_index = np.argmax(prediction)

        # ✅ Validate predicted index
        if predicted_index < 0 or predicted_index >= len(emotions):
            return jsonify({
                "error": "Invalid prediction index!",
                "index": int(predicted_index),
                "prediction": prediction.tolist()
            }), 500

        # ✅ Retrieve predicted emotion
        predicted_emotion = emotions[predicted_index]

        # ✅ Get recommended playlist (fallback to default if not found)
        recommended_playlist = emotion_playlists.get(predicted_emotion, "No playlist available")

        # ✅ Return prediction results
        return jsonify({"emotion": predicted_emotion, "playlist": recommended_playlist})

    except Exception as e:
        print("❌ Error in prediction:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500



# Run Flask Server
if __name__ == "__main__":
    app.run(debug=True)
