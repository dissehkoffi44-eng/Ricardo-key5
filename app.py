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
                bonus = (bv[i] * 0.25) + (cv[i] * 0.40) + (cv[(i+7)%12] * 0.15)
                scores[f"{NOTES_LIST[i]} {mode}"] += (corr + bonus) / len(PROFILES)
    return scores

# ────────────────────────────────────────────────
# FONCTIONS PSYCHOACOUSTIQUES AJOUTÉES
# ────────────────────────────────────────────────

def stability_bonus(key, bass_glob, chroma_glob):
    """Bonus pour la consonance de la triade de base (tonique + quinte + tierce)"""
    root_idx = NOTES_LIST.index(key.split()[0])
    mode = key.split()[1]
    
    triad_idxs = [root_idx, (root_idx + 7) % 12]
    if mode == "major":
        triad_idxs.append((root_idx + 4) % 12)
    else:
        triad_idxs.append((root_idx + 3) % 12)
    
    bass_stab = sum(bass_glob[i] for i in triad_idxs) / len(triad_idxs)
    chroma_stab = sum(chroma_glob[i] for i in triad_idxs) / len(triad_idxs)
    
    # Malus léger si sensible ou triton trop fort
    tension = bass_glob[(root_idx + 11) % 12] + bass_glob[(root_idx + 6) % 12]
    
    return 0.35 * bass_stab + 0.20 * chroma_stab - 0.08 * tension

def leading_tone_bonus(key, chroma_glob, bass_glob):
    """Bonus pour l'attraction de la sensible (leading tone)"""
    root_idx = NOTES_LIST.index(key.split()[0])
    mode = key.split()[1]
    
    if mode == "major":
        lt_idx = (root_idx + 11) % 12
        attraction = chroma_glob[lt_idx] * 0.6 + bass_glob[lt_idx] * 0.4
        return attraction * 0.18
    else:
        lt_idx = (root_idx + 10) % 12   # sensible mineure / tierce majeure
        attraction = chroma_glob[lt_idx] * 0.4 + bass_glob[lt_idx] * 0.3
        return attraction * 0.12

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
            position_ratio = start_s / duration
            # Boost fort sur la fin du morceau (résolution perçue)
            if position_ratio > 0.75:
                weight = 2.2
            elif position_ratio > 0.60:
                weight = 1.6
            elif position_ratio > 0.35:
                weight = 1.25
            else:
                weight = 0.9
            
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

    # ─── AJOUT DES BONUS PSYCHOACOUSTIQUES ───
    for key in list(final_results.keys()):
        # 1. Stabilité triadique
        stab = stability_bonus(key, bass_glob, chroma_glob)
        final_results[key] += stab * 0.22
        
        # 2. Attraction sensible / leading tone
        lt = leading_tone_bonus(key, chroma_glob, bass_glob)
        final_results[key] += lt

    # ─── Bonus triad completeness sur segments valides ───
    triad_votes = Counter()
    for k in segment_timeline:
        root_idx = NOTES_LIST.index(k.split()[0])
        mode = k.split()[1]
        triad = [root_idx, (root_idx + 7) % 12]
        if mode == "major":
            triad.append((root_idx + 4) % 12)
        else:
            triad.append((root_idx + 3) % 12)
        
        # On utilise la dernière c_seg / b_seg du segment (approximation)
        # Pour plus de précision, il faudrait stocker c_seg / b_seg par segment
        present = sum(1 for i in triad if chroma_glob[i] > 0.3 or bass_glob[i] > 0.35)
        if present >= 2:
            triad_votes[k] += 1

    for key, cnt in triad_votes.items():
        final_results[key] += (cnt / max(1, len(segment_timeline))) * 0.25

    best_key, best_score = final_results.most_common(1)[0]

    # ─── POST-PROCESSING : détection ambiguïté quinte / relative ───
    top_keys = final_results.most_common(6)

    if len(top_keys) >= 2:
        best_key_tuple, second_key_tuple = top_keys[0], top_keys[1]
        best_key, best_sc = best_key_tuple
        second_key, second_sc = second_key_tuple
        
        root1 = NOTES_LIST.index(best_key.split()[0])
        root2 = NOTES_LIST.index(second_key.split()[0])
        mode1 = best_key.split()[1]
        mode2 = second_key.split()[1]
        
        diff_semitones = (root2 - root1) % 12
        
        # Cas 1 : quinte ascendante (souvent la dominante prise pour tonique)
        if diff_semitones == 7 and mode1 == "minor" and mode2 == "minor":
            if (root1 + 7) % 12 == root2:
                st.warning(
                    f"**Cas dominant/tonic ambigu** : `{best_key}` ({CAMELOT_MAP.get(best_key)}) très fort, "
                    f"mais `{second_key}` ({CAMELOT_MAP.get(second_key)}) pourrait être la vraie tonique perçue "
                    f"(basse centrée sur la quinte). Vérifiez la résolution à la fin du morceau."
                )
                # Seuil un peu plus permissif maintenant que d'autres bonus existent
                if second_sc > 0.62 * best_sc:
                    best_key = second_key
                    best_score = second_sc

        # Cas 2 : relative major/minor confusion
        if mode1 != mode2:
            relative_diff = 3 if mode1 == "minor" else -3
            if (root1 + relative_diff) % 12 == root2:
                st.info(
                    f"**Relative possible** : `{best_key}` vs `{second_key}`. "
                    f"Dans les styles mélancoliques ou électroniques, la version mineure est souvent préférée."
                )

    return {
        "key": best_key, 
        "camelot": CAMELOT_MAP.get(best_key, "??"),
        "conf": best_score, 
        "valid_seg": valid_count, 
        "duration": duration, 
        "tuning": tuning,
        "modulation": modulation_detected,
        "top_keys": final_results.most_common(6),
        "global_scores": global_scores,
        "segment_votes_norm": segment_votes_norm,
        "segment_timeline": segment_timeline,
        "segment_timeline_counter": Counter(segment_timeline) if segment_timeline else Counter()
    }

# ────────────────────────────────────────────────
# INTERFACE (inchangée)
# ────────────────────────────────────────────────

st.title("🎵 Music Key Expert")

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
            gc.collect()

        if "error" not in data:
            data['name'] = file.name
            
            mod_text = f"\n⚠️ *Modulation détectée :* `{data['modulation']}`" if data.get('modulation') else ""
            report = (f"🎵 *{file.name}*\n"
                      f"Key: `{data['key']}` | Camelot: *{data['camelot']}*\n"
                      f"Conf: **{data['conf']*100:.1f}%** | Segments: {data['valid_seg']}"
                      f"{mod_text}")
            
            send_telegram_auto(report, bot_token, chat_id)

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

            # Top tonalités
            st.subheader("Top tonalités détectées")
            top_score = data["top_keys"][0][1] if data["top_keys"] else 1.0
            for rank, (key, score) in enumerate(data["top_keys"], 1):
                camelot = CAMELOT_MAP.get(key, "??")
                bar_length = int((score / top_score) * 40)
                st.markdown(
                    f"**{rank}.** `{key}` → **{camelot}**  \n"
                    f"Score: {score*100:.1f}%   {'█' * bar_length}"
                )

            # Global vs segments
            st.subheader("Vote global vs segments")
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Score global (70%) — Top 4")
                for k, v in sorted(data["global_scores"].items(), key=lambda x: x[1], reverse=True)[:4]:
                    st.markdown(f"`{k}` : {v*100:.1f}%")
            with col2:
                st.caption("Vote segments pondéré (30%) — Top 4")
                if data["segment_votes_norm"]:
                    for k, v in sorted(data["segment_votes_norm"].items(), key=lambda x: x[1], reverse=True)[:4]:
                        st.markdown(f"`{k}` : {v*100:.1f}%")
                else:
                    st.markdown("_Aucun segment valide_")

            # Timeline
            if data["segment_timeline"]:
                st.subheader("Évolution des tonalités par segment")
                line = [CAMELOT_MAP.get(k, "??") for k in data["segment_timeline"]]
                for chunk in range(0, len(line), 12):
                    st.code("  ".join(line[chunk:chunk+12]))
                
                st.caption("Fréquence des tonalités dans les segments valides :")
                for k, cnt in data["segment_timeline_counter"].most_common():
                    cam = CAMELOT_MAP.get(k, "??")
                    st.markdown(f"`{k}` ({cam}) : {cnt} segments")

            # Debug
            with st.expander("Debug détaillé"):
                st.markdown(f"**Durée** : {data['duration']:.1f} s")
                st.markdown(f"**Tuning** : {data['tuning']:+.2f}¢")
                st.markdown(f"**Segments valides** : {data['valid_seg']}")
                st.markdown(f"**Segments analysés** : {len(data['segment_timeline'])}")
                if data.get("modulation"):
                    st.warning(f"Modulation → {data['modulation']} ({CAMELOT_MAP.get(data['modulation'], '??')})")

        percent = i / n_files
        global_progress.progress(percent)
        global_status.text(f"{i} / {n_files} fichiers traités ({percent:.0%})")

    global_progress.progress(1.0)
    global_status.success(f"Analyse terminée — {n_files} fichier(s) traité(s)")
