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
    model = tf.keras.models.load_model("D:/BE/BE project flask api/Speech-Emotion-Recogniton/final_model.keras")
    print(model.summary())  # Debugging: Check if the model is loaded
except Exception as e:
    print("❌ Error loading model:", str(e))
    model = None

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
import soundfile as sf

import subprocess
import os
import librosa
import numpy as np
import cv2

def process_audio(file_stream):
    try:
        print("🎤 Processing in-memory audio...")

        # ✅ Save uploaded audio to a temporary file
        temp_filename = "temp_audio.wav"
        with open(temp_filename, "wb") as temp_audio:
            temp_audio.write(file_stream.read())

        # ✅ Debug: Check if file was saved properly
        if not os.path.exists(temp_filename):
            print("❌ Error: Temporary audio file not saved!")
            return None

        # ✅ Load the WAV file to check if Librosa can read it
        try:
            y, sr = librosa.load(temp_filename, sr=22050)
            print(f"🔹 Audio loaded: {len(y)} samples, Sample Rate: {sr}")
        except Exception as e:
            print(f"❌ Librosa failed to load audio: {str(e)}")
            return None

        # Ensure audio is not empty
        if y is None or len(y) == 0:
            print("❌ Error: Audio file is empty")
            return None

        # Convert to Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram = (mel_spectrogram - np.mean(mel_spectrogram)) / np.std(mel_spectrogram)

        # ✅ Resize spectrogram to match model input size
        mel_spectrogram_resized = cv2.resize(mel_spectrogram, (383, 38))  # Make sure this matches model input
        mel_spectrogram_resized = np.expand_dims(mel_spectrogram_resized, axis=(0, -1))  # Shape: (1, 383, 38, 1)

        # ✅ Cleanup temporary file
        os.remove(temp_filename)

        return mel_spectrogram_resized

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

        # ✅ Ensure processed_audio is correctly shaped
        prediction = model.predict(processed_audio)
        predicted_index = np.argmax(prediction)

        # ✅ Validate predicted_index before accessing emotions[]
        if predicted_index < 0 or predicted_index >= len(emotions):
            return jsonify({
                "error": "Invalid prediction index!",
                "index": int(predicted_index),
                "prediction": prediction.tolist()
            }), 500

        # Retrieve the predicted emotion
        predicted_emotion = emotions[predicted_index]

        # Get recommended playlist (fallback to default if not found)
        recommended_playlist = emotion_playlists.get(predicted_emotion, "No playlist available")

        # Return prediction results
        return jsonify({"emotion": predicted_emotion, "playlist": recommended_playlist})

    except Exception as e:
        print("❌ Error in prediction:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500



# Run Flask Server
if __name__ == "__main__":
    app.run(debug=True)
