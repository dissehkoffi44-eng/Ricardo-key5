import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
from scipy.signal import butter, lfilter
from scipy.io import wavfile

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 SNIPER M3 - ULTIMATE V3", page_icon="🎯", layout="wide")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_TO_SEMITONE = {n: i for i, n in enumerate(NOTES_LIST)}

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

PROFILES = {
    "shaath": {
        "major": [6.6, 2.0, 3.5, 2.3, 4.6, 4.0, 2.5, 5.2, 2.4, 3.7, 2.3, 2.9],
        "minor": [6.5, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 4.7, 4.0, 2.7, 3.3, 3.2]
    }
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-card { 
        padding: 40px; border-radius: 30px; text-align: center; color: white; 
        border: 1px solid rgba(16, 185, 129, 0.4); box-shadow: 0 15px 45px rgba(0,0,0,0.8);
        margin-bottom: 20px; background: linear-gradient(145deg, #111827, #0b0e14);
    }
    .file-header {
        background: #1f2937; color: #10b981; padding: 12px 25px; border-radius: 12px;
        font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 10px;
        border-left: 8px solid #10b981; text-transform: uppercase;
    }
    .metric-box {
        background: #161b22; border-radius: 18px; padding: 25px; text-align: center; border: 1px solid #30363d;
    }
    .verify-section {
        background: rgba(79, 70, 229, 0.1); border: 1px dashed #4F46E5;
        padding: 15px; border-radius: 15px; margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE SYNTHÈSE POUR VÉRIFICATION ---

def generate_piano_chord(key_str, duration=3.0, sr=22050):
    """ Génère un accord parfait (majeur ou mineur) pour vérification auditive """
    note_name, mode = key_str.split()
    root_idx = NOTE_TO_SEMITONE[note_name]
    
    # Intervalles d'accords (0 = tonique, 3/4 = tierce, 7 = quinte)
    intervals = [0, 4, 7] if mode == 'major' else [0, 3, 7]
    
    t = np.linspace(0, duration, int(sr * duration), False)
    chord_wave = np.zeros_like(t)
    
    for interval in intervals:
        freq = 220 * (2**((root_idx + interval) / 12)) # Octave 3 (A=220Hz)
        # Synthèse additive simple avec harmoniques pour simuler un timbre riche
        wave = np.sin(2 * np.pi * freq * t) * 0.5
        wave += np.sin(2 * np.pi * (freq * 2) * t) * 0.2 # Octave
        wave += np.sin(2 * np.pi * (freq * 3) * t) * 0.1 # Quinte harmonique
        chord_wave += wave
        
    # Enveloppe ADSR simple (fondu en sortie)
    fade_out = np.linspace(1, 0, int(sr * 0.5))
    chord_wave[-len(fade_out):] *= fade_out
    
    # Normalisation
    chord_wave = (chord_wave / np.max(np.abs(chord_wave)) * 32767).astype(np.int16)
    
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sr, chord_wave)
    return byte_io

# --- MOTEURS D'ANALYSE ---

def apply_a_weighting(y, sr):
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    a_weights = librosa.A_weighting(freqs)
    S_weighted = librosa.db_to_amplitude(librosa.amplitude_to_db(S) + a_weights[:, np.newaxis])
    return librosa.istft(S_weighted)

def get_beat_synced_chroma(y, sr, tuning):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    chroma_raw = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning, n_chroma=12, bins_per_octave=24)
    chroma_sync = librosa.util.sync(chroma_raw, beat_frames, aggregate=np.median)
    return chroma_sync, tempo, beat_frames

def solve_key_logic(chroma_vec):
    best_score = -1
    best_key = "Unknown"
    cv = (chroma_vec - np.mean(chroma_vec)) / (np.std(chroma_vec) + 1e-6)
    for mode in ["major", "minor"]:
        profile = PROFILES["shaath"][mode]
        for i in range(12):
            score = np.corrcoef(cv, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score = score
                best_key = f"{NOTES_LIST[i]} {mode}"
    return best_key, best_score

def process_audio_ultra(audio_file, file_name, placeholder):
    status = placeholder.empty()
    bar = placeholder.progress(0)

    y, sr = librosa.load(audio_file, sr=22050)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_perceptual = apply_a_weighting(y, sr)
    bar.progress(40)

    chroma_sync, tempo, beat_frames = get_beat_synced_chroma(y_perceptual, sr, tuning)
    bar.progress(70)

    timeline = []
    votes = Counter()
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    for i in range(chroma_sync.shape[1]):
        key, score = solve_key_logic(chroma_sync[:, i])
        if np.max(chroma_sync[:, i]) < 0.1: continue
        votes[key] += score
        timeline.append({"Temps": beat_times[i], "Note": key, "Conf": score})

    result_key = votes.most_common(1)[0][0]
    bar.progress(100)
    status.empty()
    bar.empty()

    return {
        "key": result_key, "camelot": CAMELOT_MAP.get(result_key),
        "conf": int(min((votes[result_key]/sum(votes.values())*100) + 30, 99)),
        "tempo": int(float(tempo)), "tuning": round(440 * (2**(tuning/12)), 1),
        "timeline": timeline, "chroma_avg": np.mean(chroma_sync, axis=1),
        "name": file_name
    }

# --- INTERFACE ---

st.title("🎯 RCDJ228 SNIPER M3 - ULTIMATE V3")
st.markdown("### Piano Verification System Enabled")

files = st.file_uploader("📂 Charger l'audio", type=['mp3','wav','flac'], accept_multiple_files=True)

if files:
    pz = st.container()
    for f in reversed(files):
        data = process_audio_ultra(f, f.name, pz)
        
        st.markdown(f"<div class='file-header'>📡 SIGNAL ANALYSÉ : {data['name']}</div>", unsafe_allow_html=True)
        
        col_main, col_side = st.columns([3, 1])
        
        with col_main:
            st.markdown(f"""
                <div class="report-card">
                    <p style="text-transform:uppercase; opacity:0.6; font-size:0.9em; letter-spacing:2px;">RÉSULTAT PERCEPTUEL</p>
                    <h1 style="font-size:6em; margin:0; color:#10b981; font-weight:900;">{data['key'].upper()}</h1>
                    <div style="font-size:1.8em; margin-bottom:15px;">CAMELOT <b>{data['camelot']}</b> • CONFIANCE <b>{data['conf']}%</b></div>
                    
                    <div class="verify-section">
                        <p style="margin-bottom:10px; font-weight:bold; color:#4F46E5;">🎹 VÉRIFICATION AUDITIVE (ACCORD DE RÉFÉRENCE)</p>
                        <p style="font-size:0.8em; opacity:0.8;">Jouez cet accord pendant que votre musique tourne pour valider la consonance.</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Génération et affichage de l'accord de vérification
            chord_audio = generate_piano_chord(data['key'])
            st.audio(chord_audio, format="audio/wav")

        with col_side:
            st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2.2em; color:#10b981;'>{data['tempo']}</span><br>BPM</div><br>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-box'><b>DIAPASON</b><br><span style='font-size:2.2em; color:#4F46E5;'>{data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)

        # Visualisations
        c1, c2 = st.columns(2)
        with c1:
            fig_tl = px.scatter(pd.DataFrame(data['timeline']), x="Temps", y="Note", color="Conf", size="Conf",
                                template="plotly_dark", title="Stabilité Harmonique", color_continuous_scale="Viridis")
            st.plotly_chart(fig_tl, use_container_width=True)
        with c2:
            fig_radar = go.Figure(data=go.Scatterpolar(r=data['chroma_avg'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
            fig_radar.update_layout(template="plotly_dark", title="Spectre de l'Empreinte", polar=dict(radialaxis=dict(visible=False)))
            st.plotly_chart(fig_radar, use_container_width=True)
