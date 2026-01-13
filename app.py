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
st.set_page_config(page_title="Audio Perception AI - Pro", layout="wide")

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
    notes_freq = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63, 'F': 349.23,
        'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
    }
    notes_list = list(notes_freq.keys())
    root_idx = notes_list.index(key)
    third_interval = 4 if tone == "Major" else 3
    third_idx = (root_idx + third_interval) % 12
    fifth_idx = (root_idx + 7) % 12
    
    frequencies = [notes_freq[key], notes_freq[notes_list[third_idx]], notes_freq[notes_list[fifth_idx]]]
    t = np.linspace(0, duration, int(sr * duration), False)
    chord_wave = np.zeros_like(t)
    
    for f in frequencies:
        chord_wave += 0.5 * np.sin(2 * np.pi * f * t)
        chord_wave += 0.25 * np.sin(2 * np.pi * (2*f) * t)
        
    envelope = np.exp(-2 * t)
    chord_wave = chord_wave * envelope
    chord_wave = (chord_wave / np.max(np.abs(chord_wave)) * 32767).astype(np.int16)
    
    byte_io = io.BytesIO()
    wav.write(byte_io, sr, chord_wave)
    return byte_io

def send_telegram_message(message):
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try: requests.post(url, json=payload)
        except: pass

@st.cache_data(show_spinner=False)
def analyze_human_perception(file_input, original_filename):
    # 1. Chargement (22kHz suffit pour l'analyse harmonique)
    y, sr = librosa.load(file_input, sr=22050)
    
    # 2. Séparation Harmonique / Percussive
    # On isole les instruments tonaux (mélodie/accords) des percussions
    y_harmonic = librosa.effects.hpss(y)[0]
    
    # 3. Chromagramme CQT (Constant-Q Transform)
    # 36 bins par octave pour une résolution fine, puis replié sur 12 demi-tons
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=36)
    
    # 4. Conversion en Décibels (Perception logarithmique humaine)
    chroma_db = librosa.amplitude_to_db(chroma, ref=np.max)
    
    # Nettoyage : on ignore le bruit sous -30dB
    chroma_db = np.maximum(chroma_db, -30)
    
    # Moyenne temporelle des notes
    chroma_vals = np.mean(chroma_db, axis=1)
    
    # Normalisation 0-1 pour la comparaison mathématique
    chroma_vals -= chroma_vals.min()
    if chroma_vals.max() > 0:
        chroma_vals /= chroma_vals.max()

    # 5. Profils de Temperley (Modernes et adaptés à la musique Pop/Rock/Electronic)
    maj_profile = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
    min_profile = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    best_score = -1
    final_key, final_tone = "", ""

    for i in range(12):
        p_maj, p_min = np.roll(maj_profile, i), np.roll(min_profile, i)
        
        # Corrélation de Pearson entre l'audio et les profils théoriques
        score_maj = np.corrcoef(chroma_vals, p_maj)[0, 1]
        score_min = np.corrcoef(chroma_vals, p_min)[0, 1]
        
        if score_maj > best_score:
            best_score, final_key, final_tone = score_maj, notes[i], "Major"
        if score_min > best_score:
            best_score, final_key, final_tone = score_min, notes[i], "Minor"

    return chroma_vals, final_key, final_tone

# --- INTERFACE ---
st.title("🧠 Perception Auditive AI Expert")
st.markdown("---")

uploaded_files = st.file_uploader("Glissez vos fichiers audio ici (MP3, WAV, FLAC)", type=["mp3", "wav", "flac"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.expander(f"🎵 Analyse : {uploaded_file.name}", expanded=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                shutil.copyfileobj(uploaded_file, tmp_file)
                tmp_path = tmp_file.name

            st.audio(uploaded_file)
            
            with st.spinner(f"Analyse perceptive de {uploaded_file.name}..."):
                try:
                    chroma_vals, key, tone = analyze_human_perception(tmp_path, uploaded_file.name)
                    camelot = get_camelot_key(key, tone)
                    result_text = f"{key} {tone}"

                    col1, col2 = st.columns(2)
                    col1.metric("Tonalité Détectée", result_text)
                    col2.metric("Code Camelot", camelot)

                    # --- SON DE VÉRIFICATION ---
                    st.write(f"### 🎹 Vérification ({result_text})")
                    chord_audio = generate_piano_chord(key, tone)
                    st.audio(chord_audio, format="audio/wav")

                    # Radar Chart des forces harmoniques
                    categories = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    fig = go.Figure(data=go.Scatterpolar(r=chroma_vals, theta=categories, fill='toself', line_color='#00FFAA'))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        template="plotly_dark",
                        margin=dict(l=50, r=50, t=20, b=20),
                        title="Empreinte Harmonique (Relative)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    send_telegram_message(f"🎵 *Analyse Experte*\n*Fichier :* {uploaded_file.name}\n*Résultat :* {result_text}\n*Camelot :* {camelot}")
                    
                except Exception as e:
                    st.error(f"Erreur sur {uploaded_file.name} : {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    st.success("Toutes les analyses sont terminées.")
