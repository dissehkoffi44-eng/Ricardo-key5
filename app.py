import streamlit as st
import librosa
import numpy as np
import requests
import os
from pydub import AudioSegment
import io
from collections import Counter

# Import CORRECT pour butter et lfilter (c'était la source de l'erreur)
from scipy.signal import butter, lfilter

st.set_page_config(page_title="Music Key & Camelot Detector", page_icon="🎵", layout="wide")

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

# Poids des différentes sources de vote (somme = 1.0)
WEIGHTS = {
    "profiles_global": 0.42,
    "segments":        0.58
}

# ────────────────────────────────────────────────
# FONCTIONS AUDIO & FILTRAGE
# ────────────────────────────────────────────────

def butter_lowpass(y, sr, cutoff=180, order=4):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, y)

def apply_sniper_filters(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=8.0)
    nyq = 0.5 * sr
    low = 60 / nyq
    high = 5000 / nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y_harm)

def get_bass_priority(y, sr):
    y_bass = butter_lowpass(y, sr, cutoff=150)
    chroma_bass = librosa.feature.chroma_cqt(y=y_bass, sr=sr, n_chroma=12)
    return np.mean(chroma_bass, axis=1)

# ────────────────────────────────────────────────
# VOTING DES PROFILS (ensemble)
# ────────────────────────────────────────────────

def vote_profiles(chroma_vector, bass_vector):
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-8)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-8)

    profile_scores = {f"{NOTES_LIST[i]} {mode}": 0.0 for i in range(12) for mode in ["major", "minor"]}

    for p_name, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]

                if mode == "minor":
                    dom_idx, leading_idx = (i + 7) % 12, (i + 11) % 12
                    if cv[dom_idx] > 0.42 and cv[leading_idx] > 0.32:
                        score *= 1.18

                if bv[i] > 0.58:
                    score += bv[i] * 0.42

                fifth_idx = (i + 7) % 12
                if cv[fifth_idx] > 0.48:
                    score += 0.14

                third_idx = (i + 4) % 12 if mode == "major" else (i + 3) % 12
                if cv[third_idx] > 0.46:
                    score += 0.10

                if cv[i] > 0.52:
                    score += 0.48

                profile_scores[f"{NOTES_LIST[i]} {mode}"] += score / len(PROFILES)

    return profile_scores

# ────────────────────────────────────────────────
# TRAITEMENT PRINCIPAL D'UN FICHIER
# ────────────────────────────────────────────────

def process_audio(file_bytes, file_name, sr_target=22050):
    ext = os.path.splitext(file_name)[1].lower()
    try:
        if ext == '.m4a':
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if audio.channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)
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
    if duration < 8:
        return {"error": "fichier trop court (< 8s)"}

    # Estimation tuning
    tuning = librosa.estimate_tuning(y=y, sr=sr)

    y_filt = apply_sniper_filters(y, sr)
    chroma_global = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning, hop_length=512), axis=1)
    bass_global = get_bass_priority(y, sr)

    # ─── VOTE 1 : profils sur global ─────────────────────────────
    global_profile_votes = vote_profiles(chroma_global, bass_global)

    # ─── VOTE 2 : analyse segmentée ──────────────────────────────
    segment_len_sec = max(10, min(30, duration / 8))
    num_segments = int(duration / segment_len_sec) + 1
    segment_votes = Counter()

    for seg in range(num_segments):
        start_sec = seg * segment_len_sec
        end_sec = min((seg + 1) * segment_len_sec, duration)
        if end_sec - start_sec < 5:
            continue

        start = int(start_sec * sr)
        end = int(end_sec * sr)
        y_seg = y_filt[start:end]
        if len(y_seg) < sr * 5 or np.max(np.abs(y_seg)) < 0.01:
            continue

        chroma_seg = np.mean(librosa.feature.chroma_cqt(y=y_seg, sr=sr, tuning=tuning, hop_length=512), axis=1)
        bass_seg = get_bass_priority(y[start:end], sr)

        seg_profile_scores = vote_profiles(chroma_seg, bass_seg)

        if seg_profile_scores:
            best_seg_key = max(seg_profile_scores, key=seg_profile_scores.get)
            seg_score = seg_profile_scores[best_seg_key]
        else:
            continue

        weight = 1.5 if 0.3 < (start_sec / duration) < 0.7 else 0.8
        segment_votes[best_seg_key] += seg_score * weight

    # Normalisation votes segments
    if segment_votes:
        total_seg = sum(segment_votes.values())
        segment_votes = {k: v / total_seg for k, v in segment_votes.items()}

    # ─── VOTE FINAL pondéré ──────────────────────────────────────
    final_votes = Counter()

    for key in set(list(global_profile_votes.keys()) + list(segment_votes.keys())):
        score = 0.0
        score += global_profile_votes.get(key, 0.0) * WEIGHTS["profiles_global"]
        score += segment_votes.get(key, 0.0) * WEIGHTS["segments"]
        final_votes[key] = score

    if not final_votes:
        return {"error": "aucune tonalité détectée"}

    best_key = final_votes.most_common(1)[0][0]
    final_conf = final_votes[best_key]

    report = f"""Analyse terminée
──────────────────────────────
Fichier       : {file_name}
Durée         : {int(duration // 60):02d}:{int(duration % 60):02d}
Fréquence     : {sr} Hz
Tuning est.   : {tuning:+.2f} cents

Tonalité      : {best_key}
Camelot       : {CAMELOT_MAP.get(best_key, "??")}
Confiance     : {final_conf:.4f}

Pondération finale : profils globaux 42% • segments 58%

Chroma global :
""" + "\n".join(f"  {k:<3} : {v:.4f}" for k,v in zip(NOTES_LIST, chroma_global))

    return {
        "key": best_key,
        "camelot": CAMELOT_MAP.get(best_key, "??"),
        "conf": final_conf,
        "report": report,
        "adjusted": False  # placeholder
    }

# ────────────────────────────────────────────────
# INTERFACE
# ────────────────────────────────────────────────

st.title("🎵 Music Key & Camelot Detector – Profiles + Segments")
st.markdown("Vote équilibré : profils globaux 42% • analyse segmentée 58% (perception supprimée)")

try:
    bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]
    chat_id   = st.secrets["TELEGRAM_CHAT_ID"]
    secrets_ok = True
except KeyError:
    bot_token = chat_id = None
    secrets_ok = False

if not secrets_ok:
    st.info("Pour activer Telegram : ajoutez TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID dans les secrets.")

uploaded_files = st.file_uploader(
    "Déposez vos fichiers audio",
    type=["mp3", "wav", "ogg", "flac", "m4a"],
    accept_multiple_files=True
)

if uploaded_files:
    total = len(uploaded_files)
    prog_global = st.progress(0)
    status_global = st.empty()

    container = st.container()

    for i, file in enumerate(uploaded_files, 1):
        prog_global.progress((i-1)/total)
        status_global.markdown(f"**Traitement {i}/{total} →** {file.name}")

        with st.status(f"Analyse → {file.name}", expanded=(i==1)) as st_status:
            st_status.write("Chargement & analyse...")
            data = process_audio(file.getvalue(), file.name)

            if "error" in data:
                st_status.update(label=f"Erreur : {data['error']}", state="error")
                continue

            st_status.update(label="Terminé ✓", state="complete", expanded=False)

            with container:
                st.markdown(f"### {file.name}")
                colA, colB = st.columns([4, 1])
                with colA:
                    st.markdown(f"**Tonalité :** {data['key']}")
                    st.markdown(f"**Camelot :** <span style='font-size:2.4em; color:#f59e0b; font-weight:bold;'>{data['camelot']}</span>", unsafe_allow_html=True)
                with colB:
                    st.metric("Confiance", f"{data['conf']:.3f}")

                st.text_area("Rapport complet", data["report"], height=380)

                if secrets_ok:
                    if st.button("Envoyer rapport Telegram", key=f"tg_{i}_{hash(file.name)}"):
                        with st.spinner("Envoi..."):
                            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                            payload = {"chat_id": chat_id, "text": f"🎵 {file.name}\n\n{data['report']}", "parse_mode": "Markdown"}
                            try:
                                r = requests.post(url, data=payload, timeout=12)
                                if r.status_code == 200:
                                    st.success("Envoyé")
                                else:
                                    st.error(f"Erreur {r.status_code}")
                            except Exception as ex:
                                st.error(f"Échec : {str(ex)}")

                st.markdown("---")

    prog_global.progress(1.0)
    status_global.success(f"✓ {total} fichier(s) analysé(s)")

st.markdown("<small>Version simplifiée : profils globaux 42% • segments 58%. Plus robuste sur EDM, rock, hip-hop moderne.</small>", unsafe_allow_html=True)
