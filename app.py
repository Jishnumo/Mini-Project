import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from python_speech_features import mfcc
import soundfile as sf
import os

# Initialize Flask App
app = Flask(__name__)

# Create uploads directory if it doesn't exist
uploads_dir = 'uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Load Dataset
def load_dataset(dataset_path):
    features = []
    labels = []
    speaker_id = 0  # Initialize speaker ID
    for speaker_folder in os.listdir(dataset_path):
        speaker_folder_path = os.path.join(dataset_path, speaker_folder)
        if os.path.isdir(speaker_folder_path):
            for file in os.listdir(speaker_folder_path):
                if file.endswith(".wav"):
                    audio_file = os.path.join(speaker_folder_path, file)
                    # Extract features and add to list
                    features.append(extract_features(audio_file))
                    labels.append(speaker_id)
            speaker_id += 1  # Increment speaker ID for the next folder
    return np.array(features), np.array(labels)

# Feature Extraction
def extract_features(audio_file, max_frames=200):
    signal, sr = sf.read(audio_file)
    mfcc_features = mfcc(signal, sr, winlen=0.025, nfft=4096)  # Increase nfft to avoid truncation
    # Ensure all features have the same length
    if mfcc_features.shape[0] < max_frames:
        mfcc_features = np.pad(mfcc_features, ((0, max_frames - mfcc_features.shape[0]), (0, 0)), mode='constant')
    elif mfcc_features.shape[0] > max_frames:
        mfcc_features = mfcc_features[:max_frames, :]
    mfcc_features = StandardScaler().fit_transform(mfcc_features)
    return mfcc_features

# Load and Split Dataset
dataset_path = "C:/Users/jishn/Desktop/MLL/16000_pcm_speeches"
X, y = load_dataset(dataset_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = SVC(kernel='linear', probability=True)
model.fit(X_train.reshape(X_train.shape[0], -1), y_train)  # Reshape X_train to 2D

# Define Speaker Names
speaker_names = {
    0: "Jishnu Mohan",
    1: "Joe Gisto",
    2: "Jishnu",
    3: "Navaneeth Krishna",
    4: "Tilin Chacko",
    5: "Naveen U",
    6: "Others",
    7: "Some kind of background noice",
    8: "TC",
}

# Define Route for Home Page
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/team')
def team():
    return render_template('team.html')
@app.route('/feature')
def feature():
    return render_template('feature.html')

# Define Route for Voice Recognition Page
@app.route('/voice')
def voice():
    return render_template('voice.html')
@app.route('/jishnu-mohan')
def jishnu():
    return render_template('jishnu-mohan.html')
@app.route('/joe-gisto')
def joe():
    return render_template('joe-gisto.html')
@app.route('/navaneeth-krishna')
def navaneeth():
    return render_template('navaneeth-krishna.html')
@app.route('/tilin-chacko')
def tilin():
    return render_template('tilin-chacko.html')
@app.route('/naveen-u')
def naveen():
    return render_template('naveen-u.html')


# Define Route for Speaker Identification
@app.route('/identify_speaker', methods=['POST'])
def identify_speaker():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    audio_file = request.files['file']
    file_path = os.path.join(uploads_dir, audio_file.filename)
    audio_file.save(file_path)

    features = extract_features(file_path)
    predicted_label = model.predict(features.reshape(1, -1))
    probability = np.max(model.predict_proba(features.reshape(1, -1)))
    speaker_id = int(predicted_label[0])  # Convert to int
    probability = float(probability)  # Convert to float
    speaker_name = speaker_names.get(speaker_id, "Unknown")  # Get speaker name from dictionary

    return jsonify({'speaker_id': speaker_id, 'speaker_name': speaker_name, 'probability': probability})

# Define Route for Downloading Speaker's Resume
@app.route('/download_resume/<string:speaker_name>')
def download_resume(speaker_name):
    # Define file locations for each resume
    resume_files = {
        "jishnu-mohan": 'C:/Users/jishn/Desktop/github/Mini Project/static/resume/Resume-Jishnu.pdf',
        "joe-gisto": 'C:/Users/jishn/Desktop/github/Mini Project/static/resume/JOEGISTO RESUME.pdf',
        "jishnu": 'C:/Users/jishn/Desktop/github/Mini Project/static/resume/resumenk.pdf',
        "navaneeth-krishna": 'C:/Users/jishn/Desktop/github/Mini Project/static/resume/resume_julia_gillard.pdf',
        "tilin-chacko": 'C:/Users/jishn/Desktop/github/Mini Project/static/resume/resume_margaret_tarcher.pdf',
        "naveen-u": 'C:/Users/jishn/Desktop/github/Mini Project/static/resume/resume_nelson_mandela.pdf',
        "others": 'C:/Users/jishn/Desktop/github/Mini Project/static/resume/resume_others.pdf',
        "some-kind-of-background-noice": 'C:/Users/jishn/Desktop/github/Mini Project/static/resume/resume_background_noise.pdf',
        # Add more entries for other speakers as needed
    }

    resume_path = resume_files.get(speaker_name)
    if resume_path:
        return send_file(resume_path, as_attachment=True)
    else:
        return jsonify({'error': 'Resume not found for the specified speaker name'})

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
