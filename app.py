import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import gc
import scipy.ndimage
from scipy.signal import butter, lfilter
import requests

# --- CONFIGURATION ---
st.set_page_config(page_title="DJ's Ear Elite v3.2", page_icon="🎧", layout="wide")

TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

# --- GÉNÉRATEUR DE TEMPLATES HARMONIQUES ---
@st.cache_resource
def get_advanced_templates():
    templates = {}
    # Profils Krumhansl-Schmuckler optimisés
    profiles = {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    }
    for mode, prof in profiles.items():
        for n in range(12):
            templates[f"{NOTES[n]} {mode}"] = np.roll(prof, n)
    return templates

# --- ANALYSE GÉOMÉTRIQUE (CYCLE DES QUINTES) ---
def circle_of_fifths_check(chroma_avg):
    """Analyse la stabilité par le voisinage des quintes."""
    fifths_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    weights = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8]
    sig = np.zeros(12)
    for i in range(12):
        # On fait pivoter le cycle pour tester chaque note comme tonique
        rotated_fifths = np.roll(fifths_order, -i)
        sig[i] = np.sum(chroma_avg[rotated_fifths] * weights)
    
    best_root = np.argmax(sig)
    # Détection de mode simple par la tierce
    is_major = chroma_avg[(best_root + 4) % 12] > chroma_avg[(best_root + 3) % 12]
    return f"{NOTES[best_root]} {'major' if is_major else 'minor'}", np.max(sig)

# --- MOTEUR DE TRAITEMENT ---
def analyze_engine_v3_2(file_bytes, file_name):
    with io.BytesIO(file_bytes) as b:
        y, sr = librosa.load(b, sr=22050)
    
    # 1. Nettoyage et Tuning
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_harm, y_perc = librosa.effects.hpss(y, margin=(8.0, 2.0))
    
    # 2. Fusion Multi-Chroma (Précision temporelle + fréquentielle)
    cqt = librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=tuning)
    cens = librosa.feature.chroma_cens(y=y_harm, sr=sr)
    fused = scipy.ndimage.median_filter(0.6 * cqt + 0.4 * cens, size=(1, 15))
    
    global_chroma = np.mean(fused, axis=1)
    templates = get_advanced_templates()
    
    # 3. Calcul du score hybride
    best_key = "C major"
    max_final_score = -1
    
    for key_name, template in templates.items():
        # Corrélation de base
        corr = np.corrcoef(global_chroma, template)[0, 1]
        
        # --- LE CORRECTIF : LE POIDS DE LA TONIQUE ---
        root_note = key_name.split()[0]
        root_idx = NOTES.index(root_note)
        
        # Boost si la note est le pic principal du chroma (ce que tu vois sur le radar)
        if np.argmax(global_chroma) == root_idx:
            corr *= 1.25 
            
        if corr > max_final_score:
            max_final_score = corr
            best_key = key_name

    # 4. Double check avec le Cercle des Quintes
    sof_key, _ = circle_of_fifths_check(global_chroma)
    
    # Si le cercle des quintes est d'accord, on renforce la certitude
    if sof_key == best_key:
        confidence = "High"
    else:
        # En cas de doute, on garde le match template mais on note l'alternative
        confidence = "Medium"

    tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
    
    return {
        "key": best_key, 
        "camelot": CAMELOT_MAP.get(best_key, "??"),
        "tempo": int(float(tempo)), 
        "tuning": round(440 * (2**(tuning/12)), 1),
        "chroma_avg": global_chroma,
        "name": file_name,
        "confidence": confidence
    }

# --- INTERFACE STREAMLIT ---
st.title("🎧 DJ's Ear Elite v3.2 (Geometric Hybrid)")

files = st.file_uploader("Upload Audio", type=['mp3','wav','flac'], accept_multiple_files=True)

if files:
    for f in reversed(files):
        with st.spinner(f"Analyse en cours... {f.name}"):
            data = analyze_engine_v3_2(f.read(), f.name)
            
        with st.expander(f"📊 {data['name']} (Confiance: {data['confidence']})", expanded=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                    <div style="background:#1e293b; padding:20px; border-radius:15px; border-left: 5px solid #3b82f6;">
                        <h2 style="color:#60a5fa; margin:0;">{data['key'].upper()}</h2>
                        <h1 style="font-size:3.5em; margin:0;">{data['camelot']}</h1>
                        <p style="opacity:0.7;">{data['tempo']} BPM | {data['tuning']} Hz</p>
                    </div>
                """, unsafe_allow_html=True)
                
                fig_polar = go.Figure(data=go.Scatterpolar(r=data['chroma_avg'], theta=NOTES, fill='toself', line_color='#60a5fa'))
                fig_polar.update_layout(template="plotly_dark", height=300, margin=dict(l=30, r=30, t=20, b=20))
                st.plotly_chart(fig_polar, use_container_width=True)
            
            with col2:
                # Visualisation du poids des notes
                df_chroma = pd.DataFrame({'Note': NOTES, 'Intensité': data['chroma_avg']})
                fig_bar = px.bar(df_chroma, x='Note', y='Intensité', title="Répartition des Énergies (Tonique)",
                                 template="plotly_dark", color='Intensité', color_continuous_scale='Blues')
                st.plotly_chart(fig_bar, use_container_width=True)
