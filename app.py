import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
import soundfile as sf
from sklearn.cluster import KMeans
import random
import os

# Helper functions
def calculate_bpm(filename):
    y, sr = librosa.load(filename, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return y, sr, tempo

def time_stretch(y, sr, original_bpm, target_bpm):
    stretch_factor = float(target_bpm) / float(original_bpm)
    y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
    return y_stretched

def convert_to_audio_segment(y, sr, filename="temp.wav"):
    sf.write(filename, y, sr)
    return AudioSegment.from_file(filename)

def crossfade_random_sections(songs, crossfade_duration=3000):
    if len(songs) == 0:
        raise ValueError("No songs provided for crossfade.")

    final_mix = songs[0]

    for i in range(1, len(songs)):
        current_song = final_mix
        next_song = songs[i]

        current_song_end = random.randint(0, len(current_song) - crossfade_duration)
        next_song_start = random.randint(0, len(next_song) - crossfade_duration)

        current_segment = current_song[:current_song_end]
        next_segment = next_song[next_song_start:]

        final_mix = current_segment.append(next_segment, crossfade=crossfade_duration)

    return final_mix

def cluster_songs(bpm_list, n_clusters=4):
    bpm_array = np.array(bpm_list).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(bpm_array)
    return clusters

# Streamlit Interface
st.title("Audio BPM Clustering and Mixing")
st.write("Upload your audio files to create a BPM-clustered mix with crossfades.")

uploaded_files = st.file_uploader("Upload Audio Files", type=["mp3", "wav"], accept_multiple_files=True)
bpm_threshold = st.number_input("BPM Threshold", min_value=1, max_value=300, value=90)
n_clusters = st.number_input("Number of Clusters", min_value=1, max_value=10, value=3)
crossfade_duration = st.slider("Crossfade Duration (ms)", min_value=1000, max_value=10000, value=3000)

if st.button("Generate Mix"):
    if uploaded_files:
        st.write("Processing uploaded files...")
        bpm_list = []
        audio_data = []

        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            y, sr, bpm = calculate_bpm(uploaded_file.name)
            st.write(f"{uploaded_file.name}: Detected BPM = {bpm}")

            if bpm >= bpm_threshold:
                bpm_list.append(bpm)
                audio_data.append((y, sr, bpm, uploaded_file.name))
            else:
                st.write(f"Excluding {uploaded_file.name} due to low BPM ({bpm} < {bpm_threshold})")

        if bpm_list:
            clusters = cluster_songs(bpm_list, n_clusters=n_clusters)
            final_mix = None

            for cluster_id in range(n_clusters):
                st.write(f"Processing cluster {cluster_id}...")
                cluster_songs = [audio_data[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
                cluster_segments = []

                target_bpm = None
                for i, (y, sr, bpm, song_path) in enumerate(cluster_songs):
                    if target_bpm is None:
                        target_bpm = bpm

                    if bpm != target_bpm:
                        y = time_stretch(y, sr, bpm, target_bpm)

                    audio_segment = convert_to_audio_segment(y, sr, filename=f"temp_cluster_{cluster_id}_song{i}.wav")
                    cluster_segments.append(audio_segment)

                if cluster_segments:
                    cluster_mix = crossfade_random_sections(cluster_segments, crossfade_duration=crossfade_duration)

                    if final_mix is None:
                        final_mix = cluster_mix
                    else:
                        final_mix = final_mix.append(cluster_mix, crossfade=crossfade_duration)

            if final_mix:
                output_file = "final_clustered_bpm_mix.mp3"
                final_mix.export(output_file, format="mp3")
                st.success(f"Final mix saved as {output_file}")

                # Stream the mix
                st.audio(output_file, format="audio/mp3")

                # Allow user to download the mix
                with open(output_file, "rb") as f:
                    st.download_button("Download Mix", f, file_name=output_file)
                os.remove(output_file)
        else:
            st.error("No songs meet the BPM threshold.")
    else:
        st.error("Please upload at least one audio file.")
