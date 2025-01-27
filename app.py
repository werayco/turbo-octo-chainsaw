import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import os
import requests
from scipy.signal import windows

# Download ffmpeg.exe
def download_ffmpeg():
    url = 'https://drive.google.com/uc?id=1w6H-h1zJ-_Ab70LaehuSo5yoqzH9PJjd&export=download'
    ffmpeg_path = "ffmpeg.exe"
    if not os.path.exists(ffmpeg_path):
        st.info("Downloading ffmpeg...")
        response = requests.get(url, stream=True)
        with open(ffmpeg_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive new chunks
                    file.write(chunk)
        st.success("ffmpeg downloaded successfully!")
    return ffmpeg_path

# Set ffmpeg path for pydub
ffmpeg_path = download_ffmpeg()
AudioSegment.converter = ffmpeg_path

# BPM-related functions
def calculate_bpm(filename):
    y, sr = librosa.load(filename, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return y, sr, tempo

def time_stretch(y, sr, original_bpm, target_bpm):
    stretch_factor = float(target_bpm) / float(original_bpm)
    if isinstance(y, np.ndarray):
        if y.ndim == 1:
            y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
        elif y.ndim == 2:
            y_stretched = np.array([librosa.effects.time_stretch(channel, rate=stretch_factor) for channel in y])
        else:
            raise ValueError("Input audio must be mono or stereo (1D or 2D array).")
    else:
        raise TypeError("Audio data (y) must be a numpy ndarray.")
    return y_stretched

def convert_to_audio_segment(y, sr, filename="temp.wav"):
    sf.write(filename, y, sr)
    return AudioSegment.from_file(filename)

def crossfade_on_bpm_threshold(song_a, song_b, bpm_a, bpm_b, crossfade_duration=3000, bpm_threshold=100):
    song_a_len = len(song_a)
    song_b_len = len(song_b)
    crossfade_start_a = int(song_a_len * 0.75)
    crossfade_start_b = int(song_b_len * 0.25)
    if bpm_a >= bpm_threshold and bpm_b >= bpm_threshold:
        segment_a = song_a[:crossfade_start_a]
        segment_b = song_b[crossfade_start_b:]
        final_mix = segment_a.append(segment_b, crossfade=crossfade_duration)
        return final_mix
    else:
        return song_a + song_b

def organize_by_bpm(audio_data):
    sorted_audio_data = sorted(audio_data, key=lambda x: x[2])
    return sorted_audio_data

# Streamlit App
st.title("BPM Based Crossfade DJ Mix Creator")
st.markdown("Upload your audio files, and generate your mix based on a fixed BPM threshold of 90.")

uploaded_files = st.file_uploader("Choose MP3 files", type=["mp3"], accept_multiple_files=True)

if uploaded_files:
    bpm_threshold = 90
    bpm_list = []
    audio_data = []

    with st.spinner("Processing your audio files..."):
        for song_file in uploaded_files:
            try:
                y, sr, bpm = calculate_bpm(song_file)
                st.write(f"Detected BPM for {song_file.name}: {bpm}")
                if bpm >= bpm_threshold:
                    bpm_list.append(bpm)
                    audio_data.append((y, sr, bpm, song_file.name))
                else:
                    st.write(f"Excluding {song_file.name} due to low BPM ({bpm} < {bpm_threshold})")
            except Exception as e:
                st.error(f"Error processing {song_file.name}: {e}")

        if not bpm_list:
            st.warning("No songs meet the BPM threshold. Exiting.")
            st.stop()

        sorted_audio_data = organize_by_bpm(audio_data)
        final_mix = None

        for i, (y, sr, bpm, song_name) in enumerate(sorted_audio_data):
            st.write(f"Processing {song_name}...")
            current_segment = convert_to_audio_segment(y, sr, filename=f"temp_song_{i}.wav")
            if i == 0:
                final_mix = current_segment
            else:
                previous_segment = final_mix
                previous_bpm = sorted_audio_data[i - 1][2]
                current_bpm = bpm
                final_mix = crossfade_on_bpm_threshold(previous_segment, current_segment, previous_bpm, current_bpm, crossfade_duration=5000, bpm_threshold=bpm_threshold)

    if final_mix:
        output_file = "final_bpm_sorted_crossfade_mix.mp3"
        final_mix.export(output_file, format="mp3")
        st.success(f"Final mix with sorted BPM and crossfades created!")
        with open(output_file, "rb") as f:
            st.download_button(
                label="Download Final Mix",
                data=f,
                file_name=output_file,
                mime="audio/mp3"
            )
        os.remove(output_file)
    else:
        st.warning("No final mix created. Check your input songs and BPM threshold.")
