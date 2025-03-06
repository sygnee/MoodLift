import os
import numpy as np
import librosa
import tensorflow as tf
import cv2
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import io
from io import BytesIO
import soundfile as sf
import bcrypt
import jwt
import datetime
from werkzeug.utils import secure_filename
import json
import pickle



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
    model = tf.keras.models.load_model("D:/BE/BE project flask api/Speech-Emotion-Recogniton/my_model.keras")
    print(model.summary())  # Debugging: Check if the model is loaded
    # Print input shape
    print("Model Input Shape:", model.input_shape)
except Exception as e:
    print("❌ Error loading model:", str(e))
    model = None




    # ✅ Load Final Emotion Mapping (43 → 7)
with open("processed_labels.pkl", "rb") as file:
    emotion_labels = pickle.load(file)  # This will store the correct labels




# Define the mapping from detailed labels to 7 core emotions
emotion_mapping = {
    "neutral": ["neutral"],
    "happy": ["happy"],
    "sad": ["sad"],
    "angry": ["angry"],
    "fear": ["fear"],
    "disgust": ["disgust"],
    "surprise": ["pleasant_surprise", "surprise"]
}

# If your labels have prefixes (e.g., "OAF_happy", "YAF_sad"), add them dynamically
for prefix in ["OAF", "YAF", "TESS"]:
    emotion_mapping[f"{prefix}_neutral"] = "neutral"
    emotion_mapping[f"{prefix}_happy"] = "happy"
    emotion_mapping[f"{prefix}_sad"] = "sad"
    emotion_mapping[f"{prefix}_angry"] = "angry"
    emotion_mapping[f"{prefix}_fear"] = "fear"
    emotion_mapping[f"{prefix}_disgust"] = "disgust"
    emotion_mapping[f"{prefix}_pleasant_surprise"] = "surprise"


# ✅ Extract 7 unique core emotions
emotions = list(emotion_mapping.keys())


# ✅ Mood-Uplifting Playlist Mapping
emotion_playlists = {
    "happy": "Feel-Good Hits 🎉 (Pop, Funk, Dance)",
    "sad": "Energy Booster 🚀 (Rock, EDM, Upbeat Pop)",
    "angry": "Chill & Relax 🧘‍♂️ (Lo-Fi, Acoustic, Jazz)",
    "fear": "Confidence Boost 💪 (Motivational Rap, Rock)",
    "disgust": "Uplifting Vibes 🌈 (Indie Pop, Soul, Funk)",
    "surprise": "Curious & Playful 🎭 (Experimental, Retro)",
    "neutral": "Positive Energy ☀️ (Indie Rock, Alternative, Happy Pop)"
}


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

        # Read audio file
        file_bytes = file_stream.read()
        file_buffer = io.BytesIO(file_bytes)
        y, sr = librosa.load(file_buffer, sr=16000)

        # Convert stereo to mono if necessary
        if len(y.shape) > 1:
            y = librosa.to_mono(y)

        # Extract 40 MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs, axis=1, keepdims=True)  # Reduce time steps

        # Expand dimensions to match model input (1, 40, 1)
        mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension

        print(f"🎤 Extracted MFCC Features: {mfccs.flatten()[:10]}")  # Print first 10 MFCC values

        #print("MFCC Mean:", np.mean(mfcc_features))
        #print("MFCC Variance:", np.var(mfcc_features))


        print(f"📏 Final Processed Audio Shape: {mfccs.shape}")  # Should print (1, 40, 1)
        return mfccs

    except Exception as e:
        print(f"❌ Error processing audio: {str(e)}")
        return None
    
#-----------------------------------------------------------------------------------------------------

# Print model output layer shape
print(f"Model Output Shape: {model.output_shape}")

# Extract labels from the final Dense layer (if available)
if hasattr(model.layers[-1], "units"):
    num_labels = model.layers[-1].units  # Get the number of output labels
    print(f"🔍 Model expects {num_labels} labels")
else:
    print("⚠️ Unable to determine number of labels from the model")


#-----------------------------------------------------------------------------------------------------

# =========================== Predict Emotion ===========================
# API Endpoint for Emotion Prediction
import io
import numpy as np
import librosa
import soundfile as sf  

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("🎤 Received request for emotion prediction")

        # ✅ Validate file upload
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided!"}), 400

        audio_file = request.files["audio"]

        # ✅ Read uploaded audio file
        audio_data, samplerate = sf.read(io.BytesIO(audio_file.read()), dtype="float32")

        # ✅ Extract MFCC features (Better normalization)
        def extract_mfcc_features(audio_signal, sr=16000, n_mfcc=40):
            mfcc = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=n_mfcc)
            mfcc = np.mean(mfcc, axis=1)  # Take mean along time axis

            # **🔹 Z-score Normalization (Improves Model Performance)**
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)  

            return mfcc.reshape(1, 40, 1)  # Reshape to (1, 40, 1)

        processed_audio = extract_mfcc_features(audio_data, samplerate)

        # 🔍 Debugging: Print MFCC values
        print("🔍 Extracted MFCC Features:", processed_audio.flatten()[:10])

        # ✅ Ensure correct shape
        if processed_audio.shape != (1, 40, 1):
            print(f"❌ Invalid Input Shape! Expected (1, 40, 1), but got {processed_audio.shape}")
            return jsonify({"error": "Processed audio shape mismatch"}), 500

        # ✅ Predict emotion
        prediction = model.predict(processed_audio)

     # ✅ Use Raw Softmax Without Temperature Scaling
        prediction = np.exp(prediction) / np.sum(np.exp(prediction), axis=-1, keepdims=True)
        prediction = prediction.flatten()

        # 🔍 Debug: Print All Probabilities
        print(f"🔍 Full Prediction Probabilities: {prediction}")
        print(f"🔍 Max Probability: {np.max(prediction):.4f}")


        # ✅ Ensure model output shape is correct
        expected_output_size = 14  
        if prediction.shape[0] != expected_output_size:
            print(f"❌ Error: Model output shape mismatch! Expected {expected_output_size}, but got {prediction.shape[0]}")
            return jsonify({"error": "Unexpected model output size"}), 500

        # ✅ Get top 3 predictions (sorted)
        top_indices = np.argsort(prediction)[-3:][::-1]
        top_emotions = [emotion_labels[idx] for idx in top_indices]
        top_scores = [prediction[idx] for idx in top_indices]

        print("🎯 Top 3 Predictions (With Scores):")
        for emo, score in zip(top_emotions, top_scores):
            print(f"{emo}: {score * 100:.2f}%")

        # ✅ Confidence-Based Adjustment
        max_confidence = np.max(prediction)
        if max_confidence > 0.98:
            print("⚠️ High confidence detected, choosing from top 3 for diversity.")
            predicted_index = np.random.choice(top_indices[:3])  
        else:
            predicted_index = top_indices[0]  # Default to best prediction

        print(f"✅ Predicted Emotion Index: {predicted_index}")

        # ✅ Retrieve detailed emotion label safely
        detailed_emotion = emotion_labels[predicted_index]  

        # ✅ Map to **7-core emotions**
        predicted_emotion = "neutral"  # Default fallback
        for core_emotion, variations in emotion_mapping.items():
            if detailed_emotion in variations or detailed_emotion.startswith(core_emotion):
                predicted_emotion = core_emotion
                break

        # ✅ Get **mood-uplifting playlist** 🎶
        uplifting_playlists = {
            "happy": "Upbeat Pop Hits",
            "neutral": "Chill Vibes",
            "sad": "Positive Energy Mix",
            "angry": "Cool Down Playlist",
            "fear": "Calm & Relaxing Sounds",
            "disgust": "Feel-Good Tracks",
            "surprise": "Exciting Beats"
        }
        recommended_playlist = uplifting_playlists.get(predicted_emotion, "No playlist available")

        return jsonify({"emotion": predicted_emotion, "playlist": recommended_playlist})

    except Exception as e:
        print("❌ Error in prediction:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500

# Run Flask Server
if __name__ == "__main__":
    app.run(debug=True)
