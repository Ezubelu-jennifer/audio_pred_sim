import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import io
import os
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model,load_model
from sklearn.preprocessing import StandardScaler
import joblib
from scipy.spatial.distance import euclidean


# model for prediction
loaded_model = joblib.load('soundlabel.pkl')

# model for similarity encoder
# Load the encoder
encoder = load_model('encoder_birdsound_model.h5')
scaler_trained = joblib.load('train_scaler.save')


def amplitude_envelope(y, frame_size, hop_length):
    return np.array([max(y[i:i+frame_size]) for i in range(0, len(y), hop_length)])


def extract_features(filepath):
    file_path = str(filepath)
    
    # Load audio file
    y, sr = librosa.load(file_path)
    
    # Estimate noise from the first 0.5 seconds
    noise_sample = y[0:int(sr * 0.5)]
    
    # Apply noise reduction
    y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
    
    # Extract features
    zero_crossings = sum(librosa.zero_crossings(y_denoised))
    spectral_centroid = librosa.feature.spectral_centroid(y=y_denoised, sr=sr).mean()
    rms = librosa.feature.rms(y=y_denoised).mean()
    peak_amplitude = max(y_denoised)
    
    frame_size = 1024
    hop_length = 512
    amp_env = amplitude_envelope(y_denoised, frame_size, hop_length)
    
    spectral_flux = librosa.onset.onset_strength(y=y_denoised, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_denoised, sr=sr).mean()
    
    # Fourier Transform for dominant frequency
    fft_spectrum = np.fft.fft(y_denoised)
    frequencies = np.fft.fftfreq(len(fft_spectrum), 1 / sr)
    magnitude = np.abs(fft_spectrum)
    dominant_freq = frequencies[np.argmax(magnitude[:len(magnitude) // 2])]
    
    # Pitch (Fundamental Frequency)
    pitches, magnitudes = librosa.piptrack(y=y_denoised, sr=sr)
    pitch = np.nanmean(pitches[magnitudes > np.median(magnitudes)]) if np.any(magnitudes) else 0
    
    # Chroma Features
    chroma = librosa.feature.chroma_stft(y=y_denoised, sr=sr).mean()
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)[0]
    
    return np.array([
    sr, 
    zero_crossings, 
    round(spectral_centroid, 2), 
    round(rms, 4), 
    round(peak_amplitude, 6), 
    round(dominant_freq, 2), 
    round(pitch, 2), 
    round(chroma, 4), 
    round(mfcc_mean, 4), 
    amp_env.mean(), 
    spectral_flux, 
    round(spectral_bandwidth, 2)
])


def predict_similarity(sound1, sound2):
    # Extract features and embeddings
    feat1 = extract_features(sound1).reshape(1, -1)
    feat2 = extract_features(sound2).reshape(1, -1)
    # Normalize the features
  
    normalized_features1 = scaler_trained.transform(feat1)
    normalized_features2 = scaler_trained.transform(feat2)


    sound1_label = loaded_model.predict(normalized_features1)
    sound2_label = loaded_model.predict(normalized_features2)
    
    # Encode both existing and new data
    encoded_existing = encoder.predict(normalized_features1)
    encoded_new = encoder.predict(normalized_features2)

   # Compute cosine similarity
    similarity_scores = cosine_similarity(encoded_existing, encoded_new)
    similarity_scores = 1-similarity_scores
    print("Similarity Score:", similarity_scores[0][0])

    distance = euclidean(encoded_existing.ravel(), encoded_new.ravel())
    print("Distance:", distance)
    

    return sound1_label, sound2_label, similarity_scores[0][0],distance

def main():
    st.title("Audio Prediction and Similarity Check")
    uploaded_file1 = st.file_uploader("Upload first audio file", type=["wav", "mp3"], key="file1")
    uploaded_file2 = st.file_uploader("Upload second audio file", type=["wav", "mp3"], key="file2")
    
    if uploaded_file1 and uploaded_file2:
        st.audio(uploaded_file1, format="audio/mp3")
        st.audio(uploaded_file2, format="audio/mp3")
        
        file_path1 = os.path.join("temp_audio", uploaded_file1.name)
        file_path2 = os.path.join("temp_audio", uploaded_file2.name)
        os.makedirs("temp_audio", exist_ok=True)
        
        with open(file_path1, "wb") as f:
            f.write(uploaded_file1.read())
        with open(file_path2, "wb") as f:
            f.write(uploaded_file2.read())
        
        label1, label2, similarity, distance = predict_similarity(file_path1, file_path2)
        
        st.write(f"Label for First Audio: {label1}")
        st.write(f"Label for Second Audio: {label2}")
        st.write(f"Similarity Score: {similarity:.4f}")
        st.write(f"Euclidean Distance: {distance:.4f}")

if __name__ == "__main__":
    main()
