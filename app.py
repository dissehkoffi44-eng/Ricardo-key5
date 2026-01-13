import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import requests
import os
import tempfile
import io
import scipy.io.wavfile as wav
import shutil

# --- CONFIGURATION ---
st.set_page_config(page_title="Audio Perception AI - HPSS 24 Bins", layout="wide")

TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

def get_camelot_key(key, tone):
    camelot_map = {
        'C Major': '8B', 'G Major': '9B', 'D Major': '10B', 'A Major': '11B', 'E Major': '12B', 'B Major': '1B',
        'F# Major': '2B', 'C# Major': '3B', 'G# Major': '4B', 'D# Major': '5B', 'A# Major': '6B', 'F Major': '7B',
        'A Minor': '8A', 'E Minor': '9A', 'B Minor': '10A', 'F# Minor': '11A', 'C# Minor': '12A', 'G# Minor': '1A',
        'D# Minor': '2A', 'A# Minor': '3A', 'F Minor': '4A', 'C Minor': '5A', 'G Minor': '6A', 'D Minor': '7A'
    }
    return camelot_map.get(f"{key} {tone}", "Inconnu")

def generate_piano_chord(key, tone, duration=2.0, sr=22050):
    notes_freq = {'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63, 'F': 349.23,
                  'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88}
    notes_list = list(notes_freq.keys())
    root_idx = notes_list.index(key)
    third_interval = 4 if tone == "Major" else 3
    frequencies = [notes_freq[key], notes_freq[notes_list[(root_idx + third_interval) % 12]], notes_freq[notes_list[(root_idx + 7) % 12]]]
    t = np.linspace(0, duration, int(sr * duration), False)
    chord_wave = np.zeros_like(t)
    for f in frequencies:
        chord_wave += 0.5 * np.sin(2 * np.pi * f * t)
    envelope = np.exp(-2 * t)
    chord_wave = (chord_wave * envelope / np.max(np.abs(chord_wave)) * 32767).astype(np.int16)
    byte_io = io.BytesIO()
    wav.write(byte_io, sr, chord_wave)
    return byte_io

@st.cache_data(show_spinner=False)
def analyze_human_perception(file_input):
    y, sr = librosa.load(file_input, sr=22050)
    
    # 1. Séparation HPSS (Harmonique / Percussif)
    # On ne garde que la partie harmonique pour éviter que la batterie ne fausse la tonalité
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # 2. Analyse CQT 24 Bins sur la partie Harmonique
    chroma_hq = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, n_chroma=24, bins_per_octave=24)
    chroma_vals_24 = np.mean(chroma_hq**2, axis=1)
    
    # 3. Analyse spécifique des BASSES (CQT sur les octaves inférieures uniquement)
    # On filtre pour ne regarder que les fréquences de 30Hz à 150Hz environ
    chroma_low = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, fmin=librosa.note_to_hz('C1'), n_octaves=3)
    bass_vals_12 = np.mean(chroma_low**2, axis=1)
    
    # Normalisation
    if np.max(chroma_vals_24) > 0: chroma_vals_24 /= np.max(chroma_vals_24)
    if np.max(bass_vals_12) > 0: bass_vals_12 /= np.max(bass_vals_12)

    # Réduction à 12 pour la corrélation (moyenne des 2 bins par note)
    chroma_vals_12 = (chroma_vals_24[0::2] + chroma_vals_24[1::2]) / 2

    # Profils Krumhansl-Schmuckler
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    best_score = -1
    final_key, final_tone = "", ""

    for i in range(12):
        p_maj, p_min = np.roll(maj_profile, i), np.roll(min_profile, i)
        
        # Corrélation harmonique globale
        score_maj = np.corrcoef(chroma_vals_12, p_maj)[0, 1]
        score_min = np.corrcoef(chroma_vals_12, p_min)[0, 1]
        
        # Pondération par les basses : si la note i est forte dans les basses, on booste le score
        # Cela aide énormément à ne pas confondre une relative mineure et sa majeure
        bass_boost = 1 + (bass_vals_12[i] * 0.5) 
        
        if (score_maj * bass_boost) > best_score:
            best_score, final_key, final_tone = score_maj * bass_boost, notes[i], "Major"
        if (score_min * bass_boost) > best_score:
            best_score, final_key, final_tone = score_min * bass_boost, notes[i], "Minor"

    return chroma_vals_24, bass_vals_12, final_key, final_tone

# --- INTERFACE ---
st.title("🧠 Audio Perception AI (HPSS + Bass Analysis)")
st.info("Cette version utilise le HPSS pour isoler les instruments mélodiques et analyse les basses pour confirmer la tonique.")

uploaded_files = st.file_uploader("Fichiers audio", type=["mp3", "wav", "flac"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.expander(f"🎵 Analyse : {uploaded_file.name}", expanded=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                shutil.copyfileobj(uploaded_file, tmp_file)
                tmp_path = tmp_file.name

            st.audio(uploaded_file)
            
            with st.spinner("Analyse harmonique et des basses..."):
                try:
                    chroma_vals, bass_vals, key, tone = analyze_human_perception(tmp_path)
                    camelot = get_camelot_key(key, tone)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tonalité", f"{key} {tone}")
                    c2.metric("Camelot", camelot)
                    c3.metric("Confiance Basse", f"{int(bass_vals[list(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']).index(key)]*100)}%")

                    st.write(f"### 🎹 Vérification ({key} {tone})")
                    st.audio(generate_piano_chord(key, tone), format="audio/wav")

                    # Radar Chart 24 Bins
                    labels_24 = []
                    for n in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']:
                        labels_24.extend([n, f"{n}½"])

                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(r=chroma_vals, theta=labels_24, fill='toself', name='Harmoniques (24 bins)', line_color='#00FFAA'))
                    fig.add_trace(go.Scatterpolar(r=bass_vals, theta=labels_24[::2], fill='toself', name='Basses (Pondération)', line_color='#FF5555'))
                    
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Erreur : {e}")
                finally:
                    if os.path.exists(tmp_path): os.remove(tmp_path)

st.success("Analyses terminées.")
