import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
import soundfile as sf
from collections import defaultdict
import random

# Harmonic keys and functions (same as the provided code)
HARMONIC_KEYS = {
    "C": ["C", "G", "Amin", "Emin"],
    "C#": ["C#", "G#", "A#min", "Fmin"],
    "D": ["D", "A", "Bmin", "F#min"],
    "D#": ["D#", "A#", "Cmin", "Gmin"],
    "E": ["E", "B", "C#min", "G#min"],
    "F": ["F", "C", "Dmin", "Amin"],
    "F#": ["F#", "C#", "D#min", "A#min"],
    "G": ["G", "D", "Emin", "Bmin"],
    "G#": ["G#", "D#", "Fmin", "Cmin"],
    "A": ["A", "E", "F#min", "C#min"],
    "A#": ["A#", "F", "Gmin", "Dmin"],
    "B": ["B", "F#", "G#min", "D#min"]
}

def add_dj_horn(audio_segment, horn_path="Dj_Air_Horn_SoundEffects.mp3"):
    horn = AudioSegment.from_file(horn_path)
    overlay_position = random.randint(5000, len(audio_segment) - 10000)  
    return audio_segment.overlay(horn, position=overlay_position)

def calculate_bpm(filename):
    y, sr = librosa.load(filename, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return y, sr, int(tempo)

def detect_key(filename):
    y, sr = librosa.load(filename, sr=None)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    pitch_class = np.argmax(chroma_mean)
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    detected_key = key_names[pitch_class]
    
    return detected_key

def time_stretch(y, sr, original_bpm, target_bpm):
    stretch_factor = target_bpm / original_bpm
    return librosa.effects.time_stretch(y, rate=stretch_factor)

def convert_to_audio_segment(y, sr, filename="temp.wav"):
    sf.write(filename, y, sr)
    return AudioSegment.from_file(filename)

def crossfade_on_bpm_threshold(song_a, song_b, bpm_a, bpm_b, crossfade_duration=3000):
    song_a_len = len(song_a)
    song_b_len = len(song_b)

    crossfade_start_a = int(song_a_len * 0.75)  
    crossfade_start_b = int(song_b_len * 0.25)  

    segment_a = song_a[:crossfade_start_a]
    segment_b = song_b[crossfade_start_b:]

    return segment_a.append(segment_b, crossfade=crossfade_duration)

def organize_by_key_and_bpm(audio_data):
    sorted_audio = defaultdict(list)

    for data in audio_data:
        key, bpm = data[3], data[2]
        sorted_audio[key].append(data)
    
    return sorted_audio

def main():
    st.title("DJ Mix Generator")
    st.write("Upload audio files and set parameters to generate your custom DJ mix.")
    
    bpm_threshold = 90
    
    uploaded_files = st.file_uploader("Upload Songs", type=["mp3"], accept_multiple_files=True)
    
    if uploaded_files:
        audio_data = []
        for uploaded_file in uploaded_files:
            song_path = uploaded_file.name
            with open(song_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            y, sr, bpm = calculate_bpm(song_path)
            key = detect_key(song_path)
            
            st.write(f"Detected BPM: {bpm}, Key: {key} for {song_path}")

            if bpm >= bpm_threshold:
                audio_data.append((y, sr, bpm, key, song_path))
            else:
                st.write(f"Skipping {song_path} due to low BPM.")

        if not audio_data:
            st.write("No valid songs found.")
            return

        sorted_audio_data = organize_by_key_and_bpm(audio_data)
        final_mix = None
        used_songs = set()

        for key, songs in sorted_audio_data.items():
            songs.sort(key=lambda x: x[2])  # Sort by BPM within the same key
            st.write(f"Mixing songs in key: {key} ({len(songs)} songs)")

            for i, (y, sr, bpm, key, song_path) in enumerate(songs):
                if song_path in used_songs:
                    continue

                current_segment = convert_to_audio_segment(y, sr, filename=f"temp_song_{i}.wav")

                if final_mix is None:
                    final_mix = current_segment
                else:
                    prev_bpm = sorted_audio_data[key][i - 1][2] if i > 0 else bpm
                    final_mix = crossfade_on_bpm_threshold(final_mix, current_segment, prev_bpm, bpm, crossfade_duration=5000)

                used_songs.add(song_path)

        if final_mix:
            final_mix = add_dj_horn(final_mix)  

            output_file = "final_harmonic_mix.mp3"
            final_mix.export(output_file, format="mp3")
            st.write(f"Final mix saved as '{output_file}' ")
            st.audio(output_file)


main()
