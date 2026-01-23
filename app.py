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
st.set_page_config(page_title="Music Key Detector - 12s Precision", page_icon="🎵", layout="wide")

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
    },
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4]
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

WEIGHTS = {
    "profiles_global": 0.65,
    "segments": 0.35
}

# ────────────────────────────────────────────────
# FONCTIONS AUDIO & FILTRAGE
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
                # Pondération interne : Corrélation + Fondamentale + Quintes
                bonus = (bv[i] * 0.45) + (cv[i] * 0.40) + (cv[(i+7)%12] * 0.15)
                scores[f"{NOTES_LIST[i]} {mode}"] += (corr + bonus) / len(PROFILES)
    return scores

# ────────────────────────────────────────────────
# TRAITEMENT PRINCIPAL
# ────────────────────────────────────────────────

def process_audio(file_bytes, file_name, sr_target=22050):
    ext = os.path.splitext(file_name)[1].lower()
    try:
        if ext == '.m4a':
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if audio.channels == 2: samples = samples.reshape(-1, 2).mean(axis=1)
            y = samples / 32768.0
            sr = audio.frame_rate
            if sr != sr_target:
                y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
                sr = sr_target
        else:
            with io.BytesIO(file_bytes) as buf:
                y, sr = librosa.load(buf, sr=sr_target, mono=True)
    except Exception as e:
        return {"error": str(e)}

    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 15: return {"error": "Fichier trop court pour l'analyse 12s"}

    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_precision_filters(y, sr)

    # 1. ANALYSE GLOBALE (Poids fort : 65%)
    chroma_glob = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)
    bass_glob = np.mean(librosa.feature.chroma_cqt(y=butter_lowpass(y, sr), sr=sr), axis=1)
    global_scores = vote_profiles(chroma_glob, bass_glob)

    # 2. ANALYSE PAR SEGMENTS (Poids : 35%)
    # Configuration demandée : 12s, Overlap 2s, Seuil 0.80
    seg_size = 12
    overlap = 2
    step = seg_size - overlap
    segment_votes = Counter()
    valid_count = 0

    for start_s in range(0, int(duration) - seg_size, step):
        start_samp = int(start_s * sr)
        end_samp = int((start_s + seg_size) * sr)
        y_seg = y_filt[start_samp : end_samp]
        
        if np.max(np.abs(y_seg)) < 0.02: continue

        c_seg = np.mean(librosa.feature.chroma_cqt(y=y_seg, sr=sr, tuning=tuning), axis=1)
        b_seg = np.mean(librosa.feature.chroma_cqt(y=butter_lowpass(y[start_samp:end_samp], sr), sr=sr), axis=1)
        
        seg_scores = vote_profiles(c_seg, b_seg)
        best_key_seg = max(seg_scores, key=seg_scores.get)
        confidence = seg_scores[best_key_seg]

        # Application du seuil de rigueur à 80%
        if confidence >= 0.80:
            # Bonus de poids au centre du morceau
            mid_weight = 1.3 if 0.25 < (start_s / duration) < 0.75 else 1.0
            segment_votes[best_key_seg] += confidence * mid_weight
            valid_count += 1

    # Normalisation des votes segments
    if segment_votes:
        total_v = sum(segment_votes.values())
        segment_votes = {k: v / total_v for k, v in segment_votes.items()}

    # COMBINAISON FINALE
    final_results = Counter()
    for key in global_scores:
        sc = (global_scores[key] * WEIGHTS["profiles_global"]) + (segment_votes.get(key, 0) * WEIGHTS["segments"])
        final_results[key] = sc

    best_key, best_score = final_results.most_common(1)[0]

    report = f"""RAPPORT DE PRÉCISION (12s Segments)
──────────────────────────────────
Fichier         : {file_name}
Tuning Estimé   : {tuning:+.2f} cents
Segments Valides: {valid_count} (Seuil > 0.80)
Poids Global    : 65%
Poids Segments  : 35%

Tonalité Finale : {best_key}
Camelot         : {CAMELOT_MAP.get(best_key, "??")}
Confiance       : {best_score:.4f}
"""
    return {
        "key": best_key,
        "camelot": CAMELOT_MAP.get(best_key, "??"),
        "conf": best_score,
        "valid_seg": valid_count,
        "report": report
    }

# ────────────────────────────────────────────────
# INTERFACE UTILISATEUR
# ────────────────────────────────────────────────

st.title("🎵 Music Key & Camelot - Precision 12s")
st.markdown("Segments : **12s** | Overlap : **2s** | Seuil de Rigueur : **80%** | Poids Global : **65%**")

# Secrets Telegram
try:
    bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]
    chat_id   = st.secrets["TELEGRAM_CHAT_ID"]
    secrets_ok = True
except:
    secrets_ok = False

uploaded_files = st.file_uploader("Glissez vos fichiers audio ici", type=["mp3", "wav", "m4a", "flac"], accept_multiple_files=True)

if uploaded_files:
    total = len(uploaded_files)
    prog_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files, 1):
        with st.status(f"Analyse en cours : {file.name}...", expanded=True) as s:
            data = process_audio(file.getvalue(), file.name)
            
            if "error" in data:
                st.error(f"Erreur : {data['error']}")
                continue
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.subheader(f"📄 {file.name}")
                st.write(f"Résultat : **{data['key']}**")
            with col2:
                st.markdown(f"<h1 style='color:#f59e0b; margin:0;'>{data['camelot']}</h1>", unsafe_allow_html=True)
            with col3:
                st.metric("Confiance", f"{data['conf']:.3f}")
                st.caption(f"{data['valid_seg']} segments certifiés")

            with st.expander("Détails de l'analyse"):
                st.text_area("Logs techniques", data["report"], height=200, key=f"log_{i}")
                if secrets_ok:
                    if st.button("Envoyer via Telegram", key=f"tg_{i}"):
                        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                        payload = {"chat_id": chat_id, "text": f"✅ *{file.name}*\n{data['report']}", "parse_mode": "Markdown"}
                        requests.post(url, data=payload)
                        st.toast("Rapport envoyé !")

            s.update(label=f"Analyse {file.name} terminée ✓", state="complete", expanded=False)
            st.divider()
        prog_bar.progress(i/total)
