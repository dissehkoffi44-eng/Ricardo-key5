import streamlit as st
import librosa
import numpy as np
import requests
import os
from pydub import AudioSegment
import io
from collections import Counter
from scipy.signal import butter, lfilter

# Configuration de la page
st.set_page_config(page_title="Music Key Detector - Precision Mode", page_icon="🎵", layout="wide")

# ────────────────────────────────────────────────
# CONSTANTES & PROFILS
# ────────────────────────────────────────────────
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

PROFILES = {
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    },
    "temperley": {
        "major": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        "minor": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
    },
    "aarden": {
        "major": [17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587, 0.291248, 22.062, 0.145624, 8.15494, 0.232998, 18.6691],
        "minor": [18.2648, 0.737619, 14.0499, 16.8599, 0.702699, 14.5212, 0.737619, 19.8145, 5.84214, 2.68046, 2.51091, 9.84455]
    }
}

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B',
    'E major': '12B', 'F major': '7B', 'F# major': '2B', 'G major': '9B',
    'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A',
    'E minor': '9A', 'F minor': '4A', 'F# minor': '11A', 'G minor': '6A',
    'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

# Pondération : Majorité au Global
WEIGHTS = {
    "profiles_global": 0.65,
    "segments": 0.35
}

# ────────────────────────────────────────────────
# FONCTIONS TECHNIQUES
# ────────────────────────────────────────────────

def butter_lowpass(y, sr, cutoff=150):
    nyq = 0.5 * sr
    b, a = butter(4, cutoff / nyq, btype='low')
    return lfilter(b, a, y)

def apply_precision_filters(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=8.0)
    nyq = 0.5 * sr
    b, a = butter(4, [60/nyq, 5000/nyq], btype='band')
    return lfilter(b, a, y_harm)

def vote_profiles(chroma_vector, bass_vector):
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-8)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-8)
    scores = {f"{n} {m}": 0.0 for n in NOTES_LIST for m in ["major", "minor"]}

    for p_data in PROFILES.values():
        for mode in ["major", "minor"]:
            for i in range(12):
                corr = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                # Bonus harmoniques (Basse, Quinte, Tierce)
                bonus = (bv[i] * 0.4) + (cv[(i+7)%12] * 0.15) + (cv[i] * 0.45)
                scores[f"{NOTES_LIST[i]} {mode}"] += (corr + bonus) / len(PROFILES)
    return scores

# ────────────────────────────────────────────────
# COEUR DE L'ANALYSE
# ────────────────────────────────────────────────

def process_audio(file_bytes, file_name):
    try:
        with io.BytesIO(file_bytes) as buf:
            y, sr = librosa.load(buf, sr=22050, mono=True)
    except:
        return {"error": "Format non supporté"}

    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_precision_filters(y, sr)

    # 1. ANALYSE GLOBALE (65%)
    chroma_glob = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)
    bass_glob = np.mean(librosa.feature.chroma_cqt(y=butter_lowpass(y, sr), sr=sr), axis=1)
    global_scores = vote_profiles(chroma_glob, bass_glob)

    # 2. ANALYSE PAR SEGMENTS (35%) - Fenêtres de 6s, Overlap 2s, Seuil 0.80
    seg_size = 6
    overlap = 2
    step = seg_size - overlap
    segment_votes = Counter()
    valid_segments = 0

    for start_s in range(0, int(duration) - seg_size, step):
        y_seg = y_filt[int(start_s * sr) : int((start_s + seg_size) * sr)]
        if np.max(np.abs(y_seg)) < 0.02: continue

        c_seg = np.mean(librosa.feature.chroma_cqt(y=y_seg, sr=sr, tuning=tuning), axis=1)
        b_seg = np.mean(librosa.feature.chroma_cqt(y=butter_lowpass(y_seg, sr), sr=sr), axis=1)
        
        seg_scores = vote_profiles(c_seg, b_seg)
        best_key = max(seg_scores, key=seg_scores.get)
        confidence = seg_scores[best_key]

        # FILTRE DE RIGUEUR : On ne prend que si c'est très sûr (> 0.80)
        if confidence >= 0.80:
            weight = 1.3 if 0.3 < (start_s / duration) < 0.7 else 1.0
            segment_votes[best_key] += confidence * weight
            valid_segments += 1

    # Normalisation segments
    if segment_votes:
        total_v = sum(segment_votes.values())
        segment_votes = {k: v / total_v for k, v in segment_votes.items()}

    # VOTE FINAL
    final_results = Counter()
    for key in global_scores:
        score = (global_scores[key] * WEIGHTS["profiles_global"]) + (segment_votes.get(key, 0) * WEIGHTS["segments"])
        final_results[key] = score

    best_final = final_results.most_common(1)[0]
    
    return {
        "key": best_final[0],
        "camelot": CAMELOT_MAP.get(best_final[0], "??"),
        "conf": best_final[1],
        "valid_seg": valid_segments,
        "report": f"Segments valides (>80%): {valid_segments}\nTuning: {tuning:+.2f}"
    }

# ────────────────────────────────────────────────
# INTERFACE
# ────────────────────────────────────────────────

st.title("🎵 Detector Pro : 6s Segments & 80% Threshold")

uploaded_files = st.file_uploader("Audios", type=["mp3", "wav", "m4a"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with st.status(f"Analyse de {file.name}...", expanded=True) as s:
            res = process_audio(file.getvalue(), file.name)
            if "error" in res:
                st.error(res["error"])
                continue
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Key", res["key"])
            col2.markdown(f"### Camelot: :orange[{res['camelot']}]")
            col3.metric("Confiance", f"{res['conf']:.3f}")
            
            st.caption(f"Analyse basée sur {res['valid_seg']} segments ultra-fiables. Global Weight: 65%.")
            s.update(label="Analyse terminée ✅", state="complete")
