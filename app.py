# RCDJ228 SNIPER M3 - EDITION "PIANO COMPANION"
# Système de décision hybride : Corrélation Statistique + Validation Théorique

import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import os
import requests
import gc
import json
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter
from datetime import datetime
from pydub import AudioSegment

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 MUSIC SNIPER PRO", page_icon="🎯", layout="wide")

# Récupération des secrets
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES & LOGIQUE PIANO COMPANION ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

# Base de données "Piano Companion" : 1 = Note autorisée dans la gamme, 0 = Note interdite
# Utilisé pour filtrer les faux positifs
COMPANION_THEORY_MASKS = {
    "major": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], # Intervalles: T, 2, 3, 4, 5, 6, 7
    "minor": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]  # Intervalles: T, 2, b3, 4, 5, b6, b7
}

PROFILES = {
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4]
    }
}

# --- FONCTIONS DE CALCUL THÉORIQUE ---

def get_theoretical_score(chroma_vector, key_name):
    """Calcule si les notes détectées matchent avec la base Piano Companion"""
    note, mode = key_name.split()
    root_idx = NOTES_LIST.index(note)
    
    # On aligne le masque théorique sur la tonique détectée
    mask = np.roll(COMPANION_THEORY_MASKS[mode], root_idx)
    
    # Score = Somme des énergies sur les notes autorisées / Énergie totale
    total_energy = np.sum(chroma_vector) + 1e-6
    allowed_energy = np.sum(chroma_vector * mask)
    
    # Pénalité si la tonique (root) ou la quinte sont absentes (Piliers Piano Companion)
    pilar_penalty = 1.0
    if chroma_vector[root_idx] < 0.4: pilar_penalty *= 0.7
    if chroma_vector[(root_idx + 7) % 12] < 0.3: pilar_penalty *= 0.8
    
    return (allowed_energy / total_energy) * pilar_penalty

def solve_key_sniper(chroma_vector, bass_vector):
    """Moteur de décision hybride"""
    best_overall_score = -1
    best_key = "Unknown"
    
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    
    for mode in ["major", "minor"]:
        for i in range(12):
            # 1. Corrélation statistique (Bellman)
            stat_score = np.corrcoef(cv, np.roll(PROFILES["bellman"][mode], i))[0, 1]
            
            # 2. Validation Piano Companion (Structure réelle)
            key_name = f"{NOTES_LIST[i]} {mode}"
            theo_fit = get_theoretical_score(cv, key_name)
            
            # Score combiné : La théorie agit comme un multiplicateur de confiance
            final_score = stat_score * (theo_fit ** 1.5)
            
            if final_score > best_overall_score:
                best_overall_score = final_score
                best_key = key_name
    
    return {"key": best_key, "score": best_overall_score}

# --- UTILITAIRES DE MIX & AUDIO ---

def get_neighbor_camelot(camelot_str: str, offset: int) -> str:
    if camelot_str in ['??', None, '']: return '??'
    try:
        num = int(camelot_str[:-1])
        wheel = camelot_str[-1]
        new_num = ((num - 1 + offset) % 12) + 1
        return f"{new_num}{wheel}"
    except: return '??'

def get_mixing_advice(data):
    if not data.get('modulation', False): return None
    principal_camelot = data.get('camelot', '??')
    target_camelot = data.get('target_camelot', '??')
    perc = data.get('mod_target_percentage', 0)
    ends_in_target = data.get('mod_ends_in_target', False)
    
    lines = ["**Checklist mix (Logique Piano Companion) :**"]
    if ends_in_target or perc > 45:
        lines.append(f"✅ Focus sur **{target_camelot}** pour la sortie.")
        priority = "target"
    else:
        lines.append(f"⚠️ Restez sur **{principal_camelot}** (modulation courte).")
        priority = "principal"
    
    lines.append(f"\n**Transition Énergie (+3) :** {get_neighbor_camelot(target_camelot if priority=='target' else principal_camelot, 3)}")
    return "\n".join(lines)

def apply_sniper_filters(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    nyq = 0.5 * sr
    b, a = butter(4, [80/nyq, 5000/nyq], btype='band')
    return lfilter(b, a, y_harm)

def seconds_to_mmss(seconds):
    if seconds is None: return "??:??"
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

# --- MOTEUR DE TRAITEMENT PRINCIPAL ---

def process_audio_precision(file_bytes, file_name, _progress_callback=None):
    try:
        with io.BytesIO(file_bytes) as buf:
            y, sr = librosa.load(buf, sr=22050, mono=True)
    except Exception as e:
        st.error(f"Erreur: {e}"); return None

    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_sniper_filters(y, sr)

    step, timeline, votes = 6, [], Counter()
    segments = list(range(0, max(1, int(duration) - step), 3))
    
    for idx, start in enumerate(segments):
        if _progress_callback: _progress_callback(int((idx/len(segments))*100), f"Analyse structurelle : {start}s")
        
        idx_s, idx_e = int(start * sr), int((start + step) * sr)
        seg = y_filt[idx_s:idx_e]
        if len(seg) < 1000 or np.max(np.abs(seg)) < 0.02: continue
        
        c_raw = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=24)
        c_avg = np.mean((c_raw[::2, :] + c_raw[1::2, :]) / 2, axis=1)
        
        # Basses isolées pour renforcement
        b_seg = np.mean(librosa.feature.chroma_cqt(y=y[idx_s:idx_e], sr=sr, fmin=librosa.note_to_hz('C1'), n_octaves=2), axis=1)
        
        res = solve_key_sniper(c_avg, b_seg)
        weight = 2.0 if (start < 15 or start > (duration - 20)) else 1.0
        votes[res['key']] += int(res['score'] * 100 * weight)
        timeline.append({"Temps": start, "Note": res['key'], "Conf": res['score']})

    if not votes: return None

    # Extraction des résultats avec validation finale
    most_common = votes.most_common(2)
    final_key = most_common[0][0]
    
    mod_detected = len(most_common) > 1 and (votes[most_common[1][0]] / sum(votes.values())) > 0.22
    target_key = most_common[1][0] if mod_detected else None

    # Calcul Tempo et Chroma Global
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma_global = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr), axis=1)

    res_obj = {
        "key": final_key, "camelot": CAMELOT_MAP.get(final_key, "??"),
        "conf": min(int(most_common[0][1]/len(timeline)), 99),
        "tempo": int(float(tempo)), "tuning": round(440 * (2**(tuning/12)), 1),
        "timeline": timeline, "chroma": chroma_global.tolist(),
        "modulation": mod_detected, "target_key": target_key,
        "target_camelot": CAMELOT_MAP.get(target_key, "??") if target_key else None,
        "mod_target_percentage": round((sum(1 for t in timeline if t["Note"] == target_key)/len(timeline))*100, 1) if mod_detected else 0,
        "mod_ends_in_target": (timeline[-1]["Note"] == target_key) if mod_detected else False,
        "modulation_time_str": seconds_to_mmss(min([t["Temps"] for t in timeline if t["Note"] == target_key])) if mod_detected else None,
        "name": file_name
    }

    # --- ENVOI TELEGRAM ---
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            caption = (f"🎯 *SNIPER REPORT (PIANO COMPANION ENGINE)*\n"
                       f"━━━━━━━━━━━━\n"
                       f"Track: `{file_name}`\n"
                       f"Key: `{res_obj['key'].upper()}` ({res_obj['camelot']})\n"
                       f"Confidence: `{res_obj['conf']}%` | Tempo: `{res_obj['tempo']} BPM`\n"
                       f"{'⚠️ MODULATION: ' + res_obj['target_key'] if mod_detected else '✅ STABLE'}")
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={'chat_id': CHAT_ID, 'text': caption, 'parse_mode': 'Markdown'})
        except: pass

    return res_obj

# --- INTERFACE STREAMLIT ---

st.markdown("<style>.metric-box { background: #161b22; border-radius: 12px; padding: 15px; text-align: center; border: 1px solid #30363d; }</style>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Fichiers Audio", type=['mp3','wav','m4a','flac'], accept_multiple_files=True)

if uploaded_files:
    for f in reversed(uploaded_files):
        with st.status(f"Sniper en action sur {f.name}...") as status:
            data = process_audio_precision(f.getvalue(), f.name, lambda v, m: status.update(label=m))
            status.update(label="Analyse certifiée terminée", state="complete")

        if data:
            st.subheader(f"🎵 {data['name']}")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"""<div style="background:linear-gradient(135deg, #1e293b, #0f172a); padding:30px; border-radius:20px; border:1px solid #3b82f6; text-align:center;">
                    <h1 style="font-size:5em; color:#fff; margin:0;">{data['key'].upper()}</h1>
                    <p style="font-size:1.5em; color:#3b82f6;">CAMELOT {data['camelot']} • CONF {data['conf']}%</p>
                </div>""", unsafe_allow_html=True)

            with col2:
                st.markdown(f"<div class='metric-box'>TEMPO<br><span style='font-size:2em; color:#10b981;'>{data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
                if data['modulation']:
                    st.warning(f"Modulation vers {data['target_key']} à {data['modulation_time_str']}")

            with col3:
                st.markdown(f"<div class='metric-box'>TUNING<br><span style='font-size:2em; color:#58a6ff;'>{data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)

            with st.expander("📊 Analyse des notes (Chroma Radar)"):
                fig_rd = go.Figure(data=go.Scatterpolar(r=data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
                fig_rd.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
                st.plotly_chart(fig_rd, use_container_width=True)

            st.divider()

with st.sidebar:
    st.title("🎯 Sniper Controls")
    if st.button("Nettoyer la session"):
        st.cache_data.clear()
        st.rerun()
