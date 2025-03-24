import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import io
import os

def amplitude_envelope(y, frame_size, hop_length):
    return np.array([max(y[i:i+frame_size]) for i in range(0, len(y), hop_length)])

def extract_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    
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

    # Return as dictionary for easier access in Streamlit UI
    return {
        "Sampling Rate": sr,
        "Zero Crossings": zero_crossings,
        "Spectral Centroid": round(spectral_centroid, 2),
        "RMS Energy": round(rms, 4),
        "Peak Amplitude": round(peak_amplitude, 6),
        "Dominant Frequency": round(dominant_freq, 2),
        "Pitch (Fundamental Frequency)": round(pitch, 2),
        "Chroma Feature": round(chroma, 4),
        "MFCC Mean": round(mfcc_mean, 4),
        "Amplitude Envelope Mean": round(amp_env.mean(), 4),
        "Spectral Flux": round(spectral_flux, 4),
        "Spectral Bandwidth": round(spectral_bandwidth, 2)
    }

# Function to apply funny filters
def apply_funny_filter(y, sr, filter_type):
    if filter_type == "Chipmunk Voice 🐿️":
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=10)  # Increase pitch
    elif filter_type == "Slow-Motion Monster 👹":
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=-8)  # Decrease pitch
    elif filter_type == "Robot Voice 🤖":
        y_mod = librosa.effects.pitch_shift(y, sr=sr, n_steps=-3)  # Deepen voice
        y_mod = librosa.effects.time_stretch(y_mod, rate=0.8)  # Slow down slightly
        return y_mod
    elif filter_type == "Echo Effect 🎤":
        # Ensure all padded arrays have the same length
        y_pad1 = np.pad(y, (10000, 0), mode="constant")[: len(y)]  # Trim to original length
        y_pad2 = np.pad(y, (5000, 0), mode="constant")[: len(y)]  
        echo = y_pad1 * 0.6 + y_pad2 * 0.4 + y * 0.8
        return echo # Keep same length
    elif filter_type == "Reverse Audio 🔄":
        return y[::-1]  # Reverse the waveform
    return y  # No filter
    

# 🎛 Streamlit UI
def main():
    st.title("🎵 Audio Feature Extraction & Modification")

    # Upload audio file
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/mp3")

        # Save file locally for processing
        file_path = os.path.join("temp_audio", uploaded_file.name)
        os.makedirs("temp_audio", exist_ok=True)  # Ensure folder exists
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extract features ✅
        features = extract_features(file_path)

        # Display extracted features 📊
        st.subheader("📊 Extracted Features")
        for key, value in features.items():
            st.write(f"**{key}:** {value}")

         
        # Load audio again for modification
        y, sr = librosa.load(file_path, sr=None)

         # 🎛 Feature Modification Sliders
        st.subheader("🎛 Modify Features & Reconstruct Audio")
        new_spectral_centroid = st.slider("Spectral Centroid", 100, 5000, int(features["Spectral Centroid"]))
        new_dominant_freq = st.slider("Dominant Frequency", 100, sr // 2, int(features["Dominant Frequency"]))
        new_rms = st.slider("RMS Energy (Loudness)", 0.0001, 0.1, float(features["RMS Energy"]))
        new_zero_crossings = st.slider("Zero Crossings", 100, 50000, int(features["Zero Crossings"]))
        new_chroma = st.slider("Chroma Feature", 0.0, 1.0, float(features["Chroma Feature"]))
        new_spectral_bandwidth = st.slider("Spectral Bandwidth", 100, 5000, int(features["Spectral Bandwidth"]))
        new_mfcc = st.slider("MFCC Mean", -500.0, 500.0, float(features["MFCC Mean"]))


        # Apply noise reduction again
        noise_sample = y[0:int(sr * 0.5)]
        y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)

        
        # Funny Filter Selection
        st.subheader("🎛 Choose a Funny Filter")
        filter_type = st.radio(
            "Select a filter:",
            ["None", "Chipmunk Voice 🐿️", "Slow-Motion Monster 👹", "Robot Voice 🤖", "Echo Effect 🎤", "Reverse Audio 🔄"],
        )

        # Apply chosen filter
        y_modified = apply_funny_filter(y_denoised, sr, filter_type)

        # 🎵 Modify Audio Based on Adjusted Features
        y_modified = librosa.effects.pitch_shift(y_modified, sr=sr, n_steps=(new_dominant_freq - features["Dominant Frequency"]) / 100)
        y_modified = y_modified * (new_rms / features["RMS Energy"])  # Adjust loudness
        y_modified = librosa.effects.time_stretch(y_modified, rate=(new_spectral_bandwidth / features["Spectral Bandwidth"]))  # Modify speed

        # Save modified audio
        modified_audio_path = os.path.join("temp_audio", "modified_audio.wav")
        sf.write(modified_audio_path, y_modified, sr, format="WAV")

        # Play modified audio
        st.subheader("🔊 Hear Your Funny Sound!")
        st.audio(modified_audio_path, format="audio/wav")
        st.download_button("Download Modified Audio", modified_audio_path, file_name="modified_audio.wav", mime="audio/wav")

if __name__ == "__main__":
    main()
