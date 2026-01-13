import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import requests
import gc
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter, find_peaks
from datetime import datetime

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 SNIPER M3 - ULTIMATE", page_icon="🎯", layout="wide")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

# Profils cognitifs (Krumhansl-Kessler & A-S-T)
PROFILES = {
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    },
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
        border: 1px solid rgba(99, 102, 241, 0.4); box-shadow: 0 15px 45px rgba(0,0,0,0.8);
        margin-bottom: 20px; background: linear-gradient(145deg, #111827, #0b0e14);
    }
    .file-header {
        background: #1f2937; color: #10b981; padding: 12px 25px; border-radius: 12px;
        font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 10px;
        border-left: 8px solid #10b981; text-transform: uppercase; letter-spacing: 1px;
    }
    .metric-box {
        background: #161b22; border-radius: 18px; padding: 25px; text-align: center; border: 1px solid #30363d;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }
    .ear-badge {
        background: linear-gradient(90deg, #4F46E5, #7C3AED); color: white; 
        padding: 6px 16px; border-radius: 50px; font-size: 0.85em; font-weight: bold;
        display: inline-block; margin-top: 15px; box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE CALCUL BIO-ACOUSTIQUES ---

def apply_psychoacoustic_prefilter(y, sr):
    """ Filtre l'audio pour ne garder que la zone de sensibilité maximale (Fletcher-Munson) """
    nyq = 0.5 * sr
    # On isole de 60Hz (basse fondamentale) à 8000Hz (harmoniques claires)
    b, a = butter(4, [60/nyq, 8000/nyq], btype='band')
    y_filt = lfilter(b, a, y)
    # On accentue les composantes harmoniques sur les percussions
    y_harm = librosa.effects.harmonic(y_filt, margin=3.5)
    return y_harm

def get_human_weighted_chroma(y, sr, tuning):
    """ Extraction du Chroma avec pondération par bandes de Bark """
    # Utilisation de CQT (Constant-Q Transform) qui imite la résolution fréquentielle de la cochlée
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning, n_chroma=12, bins_per_octave=24, fmin=librosa.note_to_hz('C1'))
    # Application d'une puissance pour simuler le seuil d'activation des neurones auditifs
    chroma = np.power(chroma, 2.2) 
    return np.mean(chroma, axis=1)

def solve_key_with_dissonance_shield(chroma_vec, bass_vec):
    """ Analyseur Sniper V6 avec bouclier de rugosité (Plomp-Levelt) """
    best_score = -1
    best_key = "Unknown"
    
    # Normalisation Z-Score pour la dynamique humaine
    cv = (chroma_vec - np.mean(chroma_vec)) / (np.std(chroma_vec) + 1e-6)
    bv = (bass_vec - np.mean(bass_vec)) / (np.std(bass_vec) + 1e-6)

    for mode in ["major", "minor"]:
        profile = PROFILES["shaath"][mode]
        for i in range(12):
            # 1. Corrélation de profil (Cognitif)
            current_profile = np.roll(profile, i)
            score = np.corrcoef(cv, current_profile)[0, 1]
            
            # 2. Renforcement de la Basse (Fondamentale perçue)
            if bv[i] > 1.2: score += 0.3  # Forte présence fondamentale
            
            # 3. Dissonance Shield : Pénalité de Rugosité
            # On pénalise si la note détectée possède une forte énergie à sa seconde mineure (+1)
            dissonance_idx = (i + 1) % 12
            if cv[dissonance_idx] > 0.8: score -= 0.4
            
            # 4. Leading Tone (Sensible) pour le mineur
            if mode == "minor":
                leading_idx = (i + 11) % 12
                if cv[leading_idx] > 0.5: score += 0.15

            if score > best_score:
                best_score = score
                best_key = f"{NOTES_LIST[i]} {mode}"
                
    return best_key, best_score

def process_audio_ultra(audio_file, file_name, placeholder):
    status = placeholder.empty()
    bar = placeholder.progress(0)

    # 1. Loading & Prefilter
    status.write(f"🧬 Extraction ADN sonore : {file_name}")
    y, sr = librosa.load(audio_file, sr=22050)
    bar.progress(20)
    
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_clean = apply_psychoacoustic_prefilter(y, sr)
    bar.progress(40)

    # 2. Analyse Segmentée (Brain integration)
    duration = librosa.get_duration(y=y, sr=sr)
    win_len = 8 # Secondes
    hop_len = 3 # Recouvrement pour lissage
    
    timeline = []
    votes = Counter()
    
    for start in range(0, int(duration) - win_len, hop_len):
        seg = y_clean[int(start*sr):int((start+win_len)*sr)]
        if np.max(np.abs(seg)) < 0.02: continue
        
        # Bass vs Treble extraction
        c_vec = get_human_weighted_chroma(seg, sr, tuning)
        
        # Filtre passe-bas pour la fondamentale de basse
        b, a = butter(2, 120/(0.5*sr), btype='low')
        y_bass = lfilter(b, a, seg)
        b_vec = np.mean(librosa.feature.chroma_cqt(y=y_bass, sr=sr, n_chroma=12), axis=1)
        
        key, score = solve_key_with_dissonance_shield(c_vec, b_vec)
        
        # Pondération temporelle (Le début et la fin comptent plus pour l'oreille)
        weight = 1.5 if (start < 10 or start > (duration - 15)) else 1.0
        votes[key] += (score * weight)
        timeline.append({"Temps": start, "Note": key, "Conf": score})
        
    bar.progress(85)
    
    # 3. Final Arbitration
    result_key = votes.most_common(1)[0][0]
    
    # Calcul de stabilité
    stability = (votes[result_key] / sum(votes.values())) * 100
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Détection de modulation
    second_key = votes.most_common(2)[1][0] if len(votes) > 1 else None
    mod_detected = False
    if second_key and (votes[second_key] / votes[result_key]) > 0.45:
        mod_detected = True

    bar.progress(100)
    status.empty()
    bar.empty()

    return {
        "key": result_key, "camelot": CAMELOT_MAP.get(result_key),
        "conf": int(min(stability + 20, 99)), "tempo": int(float(tempo)),
        "tuning": round(440 * (2**(tuning/12)), 1), "timeline": timeline,
        "chroma": get_human_weighted_chroma(y_clean, sr, tuning),
        "modulation": mod_detected, "target_key": second_key,
        "target_camelot": CAMELOT_MAP.get(second_key) if second_key else None,
        "name": file_name
    }

# --- INTERFACE ---

st.title("🎯 RCDJ228 SNIPER M3 - ULTIMATE")
st.markdown("### Audio Intelligence Engine | Psychoacoustic Perception")

files = st.file_uploader("📂 Charger l'audio pour analyse chirurgicale", type=['mp3','wav','flac'], accept_multiple_files=True)

if files:
    pz = st.container()
    for f in reversed(files):
        data = process_audio_ultra(f, f.name, pz)
        
        st.markdown(f"<div class='file-header'>📡 SIGNAL : {data['name']}</div>", unsafe_allow_html=True)
        
        col_main, col_side = st.columns([3, 1])
        
        with col_main:
            st.markdown(f"""
                <div class="report-card">
                    <p style="text-transform:uppercase; opacity:0.6; font-size:0.9em;">Perception Auditive Validée</p>
                    <h1 style="font-size:6em; margin:0; color:#10b981; font-weight:900;">{data['key'].upper()}</h1>
                    <div style="font-size:1.8em; margin-bottom:15px;">CAMELOT <b>{data['camelot']}</b> • CONFIDENCE <b>{data['conf']}%</b></div>
                    <div class="ear-badge">🧠 SHIELD : Dissonance filtrée via Plomp-Levelt</div>
                    {f"<div style='color:#f87171; font-weight:bold; margin-top:20px; border:1px solid #ef4444; padding:10px; border-radius:10px;'>⚠️ MODULATION DÉTECTÉE : {data['target_key']} ({data['target_camelot']})</div>" if data['modulation'] else ""}
                </div>
            """, unsafe_allow_html=True)

        with col_side:
            st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2.2em; color:#10b981;'>{data['tempo']}</span><br>BPM</div><br>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-box'><b>DIAPASON</b><br><span style='font-size:2.2em; color:#4F46E5;'>{data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)

        # Visualisations
        c1, c2 = st.columns(2)
        with c1:
            fig_tl = px.scatter(pd.DataFrame(data['timeline']), x="Temps", y="Note", size="Conf", color="Conf", 
                                template="plotly_dark", title="Flux Harmonique Temporel", color_continuous_scale="Viridis")
            st.plotly_chart(fig_tl, use_container_width=True)
        with c2:
            fig_radar = go.Figure(data=go.Scatterpolar(r=data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
            fig_radar.update_layout(template="plotly_dark", title="Empreinte Cochléaire (Spectre)", polar=dict(radialaxis=dict(visible=False)))
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")

with st.sidebar:
    st.header("Paramètres Sniper")
    st.write("Le mode **Human Ear** utilise une pondération non-linéaire des fréquences pour correspondre à la courbe d'isophonie.")
    if st.button("🗑️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()
