document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.getElementById("loginForm");
    const signupForm = document.getElementById("signupForm");
    const recordBtn = document.getElementById("record-btn");
    const stopBtn = document.getElementById("stop-btn");
    const emotionResult = document.getElementById("emotion-result");
    const playlistContainer = document.getElementById("playlist-container");
    const toggleDarkMode = document.getElementById("toggle-darkmode");

    //  Declare Global Variables ONCE
    var mediaRecorder = null;  //  Use `var` instead of `let` to avoid redeclaration error
    var audioChunks = [];
    var stream = null; //  Keep track of microphone stream

    let mediaRecorder = null;
    let audioChunks = [];
    let stream = null; //  Added to track mic usage globally

    

      // Dark Mode Toggle
      if (toggleDarkMode) {
        toggleDarkMode.addEventListener("click", () => {
            document.body.classList.toggle("dark-mode");
        });
    }

    // User Authentication
    async function authenticateUser(url, credentials) {
        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(credentials)
        });
    
        const data = await response.json();
        console.log("🔹 Auth Response:", data); // Debugging log
    
        if (response.ok) {
            localStorage.setItem("token", data.token);
            localStorage.setItem("username", credentials.username);
            document.getElementById("authSection").classList.add("hidden");
            document.getElementById("mainApp").classList.remove("hidden");
            document.getElementById("userSection").classList.remove("hidden");
            document.getElementById("userName").textContent = credentials.username;

        } else {
            alert(data.error);
        }
    }
    
    

    if (loginForm) {
        loginForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            console.log("🔹 Login button clicked!");
    
            const username = document.getElementById("username").value.toLowerCase(); // Case-insensitive login
            const password = document.getElementById("password").value;
    
            const response = await fetch("http://127.0.0.1:5000/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, password })
            });
    
            const data = await response.json();
            console.log(" Login Response:", data);
    
            if (data.error) {
                alert(` Error: ${data.error}`);
            } else {
                localStorage.setItem("token", data.token);
                alert("Login successful!");
                window.location.href = "index.html";  // Redirect to login page after signup

            }
        });
    }
    
    

    if (signupForm) {
        signupForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            console.log("🔹 Signup button clicked!");
    
            const username = document.getElementById("new-username").value;
            const password = document.getElementById("new-password").value;
            const confirmPassword = document.getElementById("confirm-password").value;
    
            if (password !== confirmPassword) {
                alert("Passwords do not match!");
                return;
            }
    
            console.log(" Sending signup request...");
    
            const response = await fetch("http://127.0.0.1:5000/signup", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, password })
            });
    
            const data = await response.json();
            console.log(" Signup Response:", data);
    
            if (data.error) {
                alert(`Error: ${data.error}`);
            } else {
                alert(" Account created successfully! Please log in.");
                window.location.href = "index.html";  // Redirect to main login page

            }
        });
    }
    
    

    // Voice Recording and Emotion Detection
    let mediaRecorder;
    let audioChunks = [];
    let stream; //  Store mic stream globally so we can stop it properly

    if (recordBtn) {
        recordBtn.addEventListener("click", async () => {
            try {
                // If mic is already open, stop it before starting a new one
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
    
                // Open a new mic stream
                stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = []; // Reset audio chunks for new recording
    
                mediaRecorder.start();
                recordBtn.disabled = true;
                stopBtn.disabled = false;
    
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
    
                console.log(" Recording started...");
            } catch (error) {
                alert(" Error accessing microphone: " + error.message);
                console.error("Mic Error:", error);
            }
        });
    }
    
    if (stopBtn) {
        stopBtn.addEventListener("click", async () => {
            if (!mediaRecorder || mediaRecorder.state !== "recording") {
                alert("No active recording found.");
                return;
            }
    
            mediaRecorder.stop();
            recordBtn.disabled = false;
            stopBtn.disabled = true;
    
            mediaRecorder.onstop = async () => {
                console.log(" Recording stopped");
    
                if (audioChunks.length === 0) {
                    alert(" No audio recorded. Please try again.");
                    return;
                }
    
                const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
                audioChunks = []; //  Clear old recordings to avoid conflicts
    
                const formData = new FormData();
                formData.append("audio", audioBlob, "recording.webm");
    
                try {
                    const response = await fetch("http://127.0.0.1:5000/predict", {
                        method: "POST",
                        body: formData
                    });
    
                    const data = await response.json();
                    console.log(" Emotion Prediction Response:", data);
    
                    if (response.ok) {
                        emotionResult.innerHTML = `Detected Emotion: <span>${data.emotion}</span>`;
                        displayMoodBoostingPlaylists(data.emotion);
                    } else {
                        alert(` Error: ${data.error}`);
                    }
                } catch (error) {
                    alert(" Network error. Please check your connection.");
                    console.error("Fetch error:", error);
                }
    
                // FIX: Stop microphone access properly
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    console.log(" Microphone access stopped");
                    stream = null; //  Reset stream variable to ensure a fresh one is created next time
                }
            };
        });
    }
    
    
    

    // Fetch and Display Mood-Boosting Playlists from Spotify
    // Fetch and Display Mood-Boosting Playlists from Spotify
    function displayMoodBoostingPlaylists(emotion) {
        playlistContainer.innerHTML = "<p>Loading playlists...</p>";
        const moodPlaylists = {
            "neutral": [
                { name: "Good Vibes", url: "https://open.spotify.com/playlist/37i9dQZF1DX6RV1avkEs6h" }
            ],
            "calm": [
                { name: "Peaceful Piano", url: "https://open.spotify.com/playlist/37i9dQZF1DX9sIqqvKsjG8" }
            ],
            "happy": [
                { name: "Upbeat Energy", url: "https://open.spotify.com/playlist/37i9dQZF1DX4fpCWaHOned" }
            ],
            "sad": [
                { name: "Cheer Up!", url: "https://open.spotify.com/playlist/37i9dQZF1DWZjqjZMudx9T" }
            ],
            "angry": [
                { name: "Relax & Unwind", url: "https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO" }
            ],
            "fear": [
                { name: "Confidence Boost", url: "https://open.spotify.com/playlist/37i9dQZF1DX4fpCWaHOned" }
            ],
            "disgust": [
                { name: "Feel-Good Classics", url: "https://open.spotify.com/playlist/37i9dQZF1DXb5Mq0JeBbIw" }
            ],
            "surprise": [
                { name: "Exciting Beats", url: "https://open.spotify.com/playlist/37i9dQZF1DX0XUsuxWHRQd" }
            ]
        };
        
        
        
        const selectedPlaylists = moodPlaylists[emotion] || moodPlaylists["neutral"];
        playlistContainer.innerHTML = "";
        selectedPlaylists.forEach(playlist => {
            const playlistElement = document.createElement("a");
            playlistElement.href = playlist.url;
            playlistElement.textContent = ` ${playlist.name}`;
            playlistElement.target = "_blank";
            playlistElement.classList.add("playlist-item");
            playlistContainer.appendChild(playlistElement);
        });
    }
});

