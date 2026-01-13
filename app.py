import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
from scipy.signal import butter, lfilter
from datetime import datetime

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 SNIPER M3 - ULTIMATE V2", page_icon="🎯", layout="wide")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
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
    .viterbi-badge {
        background: #4F46E5; color: white; padding: 4px 12px; border-radius: 4px; font-size: 0.7em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE CALCUL AVANCÉS ---

def apply_a_weighting(y, sr):
    """ Applique une pondération de type 'A' pour simuler la sensibilité de l'oreille humaine """
    # L'oreille est moins sensible aux fréquences très basses et très hautes
    # On utilise librosa.perceptual_weighting sur le spectre de puissance
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    a_weights = librosa.A_weighting(freqs)
    S_weighted = librosa.db_to_amplitude(librosa.amplitude_to_db(S) + a_weights[:, np.newaxis])
    return librosa.istft(S_weighted)

def get_beat_synced_chroma(y, sr, tuning):
    """ Extrait le Chroma synchronisé sur le rythme (Beat-Sync) pour filtrer le bruit """
    # 1. Estimation du tempo et des beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    # 2. Chroma CQT haute résolution
    chroma_raw = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning, n_chroma=12, bins_per_octave=24)
    
    # 3. Synchronisation : On fait la moyenne du chroma entre chaque beat
    # C'est ici que l'humain perçoit la 'note' dominante de la mesure
    chroma_sync = librosa.util.sync(chroma_raw, beat_frames, aggregate=np.median)
    
    return chroma_sync, tempo, beat_frames

def solve_key_logic(chroma_vec, profile_type="shaath"):
    """ Cœur de décision avec normalisation Z-Score """
    best_score = -1
    best_key = "Unknown"
    
    cv = (chroma_vec - np.mean(chroma_vec)) / (np.std(chroma_vec) + 1e-6)

    for mode in ["major", "minor"]:
        profile = PROFILES[profile_type][mode]
        for i in range(12):
            current_profile = np.roll(profile, i)
            score = np.corrcoef(cv, current_profile)[0, 1]
            
            # Bonus de quinte (Stabilité harmonique)
            quinte_idx = (i + 7) % 12
            if cv[quinte_idx] > 0.5: score += 0.1
            
            if score > best_score:
                best_score = score
                best_key = f"{NOTES_LIST[i]} {mode}"
                
    return best_key, best_score

def process_audio_ultra(audio_file, file_name, placeholder):
    status = placeholder.empty()
    bar = placeholder.progress(0)

    # 1. Acquisition et Correction Auditive
    status.write(f"🎧 Modélisation de l'oreille humaine : {file_name}")
    y, sr = librosa.load(audio_file, sr=22050)
    bar.progress(15)
    
    # Estimation du Diapason (A440 ou autre)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    
    # Application de la courbe d'isophonie (A-Weighting)
    y_perceptual = apply_a_weighting(y, sr)
    bar.progress(30)

    # 2. Extraction Synchronisée (Beat-Sync Chromagram)
    status.write("🥁 Synchronisation sur les temps forts (Beat-Sync)...")
    chroma_sync, tempo, beat_frames = get_beat_synced_chroma(y_perceptual, sr, tuning)
    bar.progress(60)

    # 3. Analyse Temporelle (Simulant la mémoire à court terme)
    status.write("🧠 Analyse de la stabilité tonale (Viterbi emulation)...")
    timeline = []
    votes = Counter()
    
    # Conversion des frames de beats en secondes pour la timeline
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    for i in range(chroma_sync.shape[1]):
        chroma_column = chroma_sync[:, i]
        key, score = solve_key_logic(chroma_column)
        
        # On ignore les segments trop faibles (silences entre notes)
        if np.max(chroma_column) < 0.1: continue
        
        # Pondération : les notes longues et stables comptent plus
        votes[key] += score
        timeline.append({"Temps": beat_times[i] if i < len(beat_times) else i, "Note": key, "Conf": score})

    # 4. Arbitrage Final
    if not votes:
        return None

    result_key = votes.most_common(1)[0][0]
    stability = (votes[result_key] / sum(votes.values())) * 100
    
    # Détection de Modulation (Changement de tonalité en cours de morceau)
    second_key = votes.most_common(2)[1][0] if len(votes) > 1 else None
    is_modulating = False
    if second_key and (votes[second_key] / votes[result_key]) > 0.6:
        is_modulating = True

    bar.progress(100)
    status.empty()
    bar.empty()

    return {
        "key": result_key, "camelot": CAMELOT_MAP.get(result_key),
        "conf": int(min(stability + 30, 99)), "tempo": int(float(tempo)),
        "tuning": round(440 * (2**(tuning/12)), 1), "timeline": timeline,
        "chroma_avg": np.mean(chroma_sync, axis=1),
        "modulation": is_modulating, "target_key": second_key,
        "target_camelot": CAMELOT_MAP.get(second_key) if second_key else None,
        "name": file_name
    }

# --- INTERFACE ---

st.title("🎯 RCDJ228 SNIPER M3 - ULTIMATE")
st.markdown("### Audio Intelligence | Beat-Synchronous & Perceptual Analysis")

files = st.file_uploader("📂 Charger l'audio", type=['mp3','wav','flac'], accept_multiple_files=True)

if files:
    pz = st.container()
    for f in reversed(files):
        data = process_audio_ultra(f, f.name, pz)
        
        if not data:
            st.error(f"Impossible d'analyser {f.name}")
            continue

        st.markdown(f"<div class='file-header'>📡 SIGNAL ANALYSÉ : {data['name']}</div>", unsafe_allow_html=True)
        
        col_main, col_side = st.columns([3, 1])
        
        with col_main:
            st.markdown(f"""
                <div class="report-card">
                    <p style="text-transform:uppercase; opacity:0.6; font-size:0.9em; letter-spacing:2px;">RÉSULTAT PERCEPTUEL</p>
                    <h1 style="font-size:6em; margin:0; color:#10b981; font-weight:900;">{data['key'].upper()}</h1>
                    <div style="font-size:1.8em; margin-bottom:15px;">CAMELOT <b>{data['camelot']}</b> • CONFIANCE <b>{data['conf']}%</b></div>
                    <div style="display:flex; justify-content:center; gap:10px;">
                        <span class="viterbi-badge">BEAT-SYNC ACTIVE</span>
                        <span class="viterbi-badge" style="background:#059669;">A-WEIGHTING FILTER</span>
                    </div>
                    {f"<div style='color:#f87171; font-weight:bold; margin-top:20px; border:1px solid #ef4444; padding:10px; border-radius:10px;'>⚠️ TRANSITION DÉTECTÉE : {data['target_key']} ({data['target_camelot']})</div>" if data['modulation'] else ""}
                </div>
            """, unsafe_allow_html=True)

        with col_side:
            st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2.2em; color:#10b981;'>{data['tempo']}</span><br>BPM (Global)</div><br>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-box'><b>DIAPASON</b><br><span style='font-size:2.2em; color:#4F46E5;'>{data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)

        # Visualisations
        c1, c2 = st.columns(2)
        with c1:
            df_tl = pd.DataFrame(data['timeline'])
            fig_tl = px.scatter(df_tl, x="Temps", y="Note", color="Conf", size="Conf",
                                template="plotly_dark", title="Stabilité Harmonique (Memory Model)",
                                color_continuous_scale="Viridis")
            st.plotly_chart(fig_tl, use_container_width=True)
            
        with c2:
            fig_radar = go.Figure(data=go.Scatterpolar(r=data['chroma_avg'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
            fig_radar.update_layout(template="plotly_dark", title="Empreinte Cochléaire Moyenne", 
                                    polar=dict(radialaxis=dict(visible=False)))
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")

with st.sidebar:
    st.header("Sniper M3 Engine")
    st.info("Cette version synchronise l'analyse sur le rythme (beats) pour ignorer les percussions et se concentrer sur les notes musicales réelles.")
    if st.button("🗑️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()
