import streamlit as st
import librosa
import numpy as np
import requests
import os
from pydub import AudioSegment
import io
from collections import Counter
from scipy.signal import butter, lfilter
import gc

# Configuration de la page
st.set_page_config(page_title="Music Key Expert", page_icon="🎵", layout="wide")

# --- FORCE FFMPEG PATH (Optionnel) ---
if os.path.exists(r'C:\ffmpeg\bin'):
    os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'

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

WEIGHTS = {"profiles_global": 0.70, "segments": 0.30}

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

def send_telegram_auto(msg, bot_token, chat_id):
    if bot_token and chat_id:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
        try:
            requests.post(url, data=payload, timeout=10)
        except:
            pass

def vote_profiles(chroma_vector, bass_vector):
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-8)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-8)
    scores = {f"{n} {m}": 0.0 for n in NOTES_LIST for m in ["major", "minor"]}
    for p_data in PROFILES.values():
        for mode in ["major", "minor"]:
            for i in range(12):
                corr = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                bonus = (bv[i] * 0.45) + (cv[i] * 0.40) + (cv[(i+7)%12] * 0.15)
                scores[f"{NOTES_LIST[i]} {mode}"] += (corr + bonus) / len(PROFILES)
    return scores

# ────────────────────────────────────────────────
# TRAITEMENT UNIFIÉ AVEC DÉTECTION DE MODULATION
# ────────────────────────────────────────────────

def process_audio(file_bytes, file_name, sr_target=22050):
    ext = os.path.splitext(file_name)[1].lower()
    try:
        if ext == '.m4a':
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if audio.channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)
            y = samples / (2**(8 * audio.sample_width - 1))
            sr = audio.frame_rate
            if sr != sr_target:
                y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
                sr = sr_target
        else:
            with io.BytesIO(file_bytes) as buf:
                y, sr = librosa.load(buf, sr=sr_target, mono=True)

    except Exception as e:
        return {"error": f"Erreur de décodage ({ext}): {str(e)}"}

    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_precision_filters(y, sr)

    # Analyse globale
    chroma_glob = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)
    bass_glob = np.mean(librosa.feature.chroma_cqt(y=butter_lowpass(y, sr), sr=sr), axis=1)
    global_scores = vote_profiles(chroma_glob, bass_glob)

    # Analyse par segments
    seg_size, overlap = 12, 6
    step = seg_size - overlap
    segment_votes = Counter()
    segment_timeline = []
    valid_count = 0

    for start_s in range(0, int(duration) - seg_size, step):
        y_seg = y_filt[int(start_s * sr) : int((start_s + seg_size) * sr)]
        if np.max(np.abs(y_seg)) < 0.02: continue
        
        c_seg = np.mean(librosa.feature.chroma_cqt(y=y_seg, sr=sr, tuning=tuning), axis=1)
        b_seg = np.mean(librosa.feature.chroma_cqt(y=butter_lowpass(y_seg, sr), sr=sr), axis=1)
        
        seg_scores = vote_profiles(c_seg, b_seg)
        best_k = max(seg_scores, key=seg_scores.get)
        
        if seg_scores[best_k] >= 0.75:
            weight = 1.3 if 0.25 < (start_s / duration) < 0.75 else 1.0
            segment_votes[best_k] += seg_scores[best_k] * weight
            segment_timeline.append(best_k)
            valid_count += 1

    # Détection modulation
    modulation_detected = None
    if len(segment_timeline) >= 4:
        mid = len(segment_timeline) // 2
        first_half_key = Counter(segment_timeline[:mid]).most_common(1)[0][0]
        second_half_key = Counter(segment_timeline[mid:]).most_common(1)[0][0]
        if first_half_key != second_half_key:
            modulation_detected = second_half_key

    # Score final pondéré
    if segment_votes:
        total_v = sum(segment_votes.values())
        segment_votes_norm = {k: v / total_v for k, v in segment_votes.items()}
    else:
        segment_votes_norm = {}

    final_results = Counter()
    for key in global_scores:
        final_results[key] = (global_scores[key] * WEIGHTS["profiles_global"]) + (segment_votes_norm.get(key, 0) * WEIGHTS["segments"])

    best_key, best_score = final_results.most_common(1)[0]
    
    return {
        "key": best_key, 
        "camelot": CAMELOT_MAP.get(best_key, "??"),
        "conf": best_score, 
        "valid_seg": valid_count, 
        "duration": duration, 
        "tuning": tuning,
        "modulation": modulation_detected
    }

# ────────────────────────────────────────────────
# INTERFACE
# ────────────────────────────────────────────────

st.title("🎵 Universal Key Detector (Modulation Aware)")

# Barre de progression globale
global_progress = st.progress(0)
global_status = st.empty()

bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN")
chat_id = st.secrets.get("TELEGRAM_CHAT_ID")

uploaded_files = st.file_uploader("Audios (FLAC, MP3, WAV, M4A)", type=["flac", "mp3", "wav", "m4a"], accept_multiple_files=True)

if uploaded_files:
    n_files = len(uploaded_files)

    global_progress.progress(0)
    global_status.text(f"0 / {n_files} fichiers traités (0%)")

    for i, file in enumerate(uploaded_files, 1):
        percent = (i - 1) / n_files
        global_progress.progress(percent)
        global_status.text(f"{i-1} / {n_files} fichiers traités ({percent:.0%})")

        with st.spinner(f"Analyse de {file.name} ({i}/{n_files})"):
            data = process_audio(file.getvalue(), file.name)
            gc.collect()  # Libération mémoire après chaque fichier

        if "error" not in data:
            data['name'] = file.name
            
            mod_text = f"\n⚠️ *Modulation détectée :* `{data['modulation']}`" if data.get('modulation') else ""
            report = (f"🎵 *{file.name}*\n"
                      f"Key: `{data['key']}` | Camelot: *{data['camelot']}*\n"
                      f"Conf: **{data['conf']*100:.1f}%** | Segments: {data['valid_seg']}"
                      f"{mod_text}")
            
            send_telegram_auto(report, bot_token, chat_id)

            # Affichage immédiat du résultat
            st.markdown("---")
            with st.container():
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                with c1:
                    st.markdown(f"**{data['name']}**")
                    st.caption(f"Format: {data['name'].split('.')[-1].upper()}  |  Tuning: {data['tuning']:+.2f}¢")
                with c2:
                    st.markdown(f"<h2 style='color:#f59e0b; margin:0; text-align:center;'>{data['camelot']}</h2>", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"**{data['key']}**")
                    if data.get('modulation'):
                        mod_cam = CAMELOT_MAP.get(data['modulation'], "??")
                        st.markdown(f"<p style='color:#ef4444; font-size:0.85em; margin-top:6px;'>→ Mod: {data['modulation']} ({mod_cam})</p>", unsafe_allow_html=True)
                with c4:
                    st.metric("Confiance", f"{data['conf']*100:.1f} %")
                st.divider()

        percent = i / n_files
        global_progress.progress(percent)
        global_status.text(f"{i} / {n_files} fichiers traités ({percent:.0%})")

    global_progress.progress(1.0)
    global_status.success(f"Analyse terminée — {n_files} fichier(s) traité(s)")
