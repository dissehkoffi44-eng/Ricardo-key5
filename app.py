import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import io
import gc
import scipy.ndimage
from scipy.signal import butter, lfilter
import requests

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="L'Elite", page_icon="🎧", layout="wide")

# --- GESTION DES SECRETS ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

# --- GÉNÉRATION DE TEMPLATES (Niveau 1 : plus d'harmoniques + décroissance réaliste) ---
@st.cache_resource
def generate_real_templates(sr=22050, A4=440.0, duration=1.0):
    templates = {}
    harmonic_weights = [1.00, 0.50, 0.30, 0.15, 0.08]  # Fondamental + 4 harmoniques décroissants

    for mode in ["major", "minor"]:
        intervals = [0, 4, 7] if mode == "major" else [0, 3, 7]
        for i, root in enumerate(NOTES):
            freqs = []
            for intv in intervals:
                note_num = i + intv
                freq = A4 * (2 ** ((note_num - 9) / 12))
                freqs.append(freq)
            
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            y = np.zeros_like(t)
            for harm_idx, weight in enumerate(harmonic_weights):
                for f in freqs:
                    y += weight * np.sin(2 * np.pi * (harm_idx + 1) * f * t)
            
            y = librosa.util.normalize(y)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
            chroma_avg = np.mean(chroma, axis=1)
            chroma_avg = (chroma_avg - np.mean(chroma_avg)) / (np.std(chroma_avg) + 1e-8)
            templates[f"{root} {mode}"] = chroma_avg
    return templates

# --- Signature des quintes ---
def signature_of_fifths_key(chroma_avg):
    fifths_order = [0,7,2,9,4,11,6,1,8,3,10,5]
    weights = [1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04]
    sig = np.zeros(12)
    for i in range(12):
        rolled = (np.array(fifths_order) + i) % 12
        sig[i] = np.sum(chroma_avg[rolled] * np.array(weights))
    
    best_root = np.argmax(sig)
    third_maj = (best_root + 4) % 12
    third_min = (best_root + 3) % 12
    mode = "major" if chroma_avg[third_maj] > chroma_avg[third_min] else "minor"
    return f"{NOTES[best_root]} {mode}", np.max(sig)

# --- FONCTION D'ENVOI TELEGRAM ---
def send_telegram_expert(data, fig_timeline, fig_radar):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return

    msg = (f" *L'Elite*\n"
           f"━━━━━━━━━━━━━━━━━━━━\n"
           f" *Fichier:* `{data['name']}`\n\n"
           f" *TONALITÉ PRINCIPALE*\n"
           f"└ Note : `{data['key'].upper()}`\n"
           f"└ Camelot : `{data['camelot']}`\n\n"
           f" *MÉTRIQUES*\n"
           f"└ Tempo : `{data['tempo']} BPM`\n"
           f"└ Tuning : `{data['tuning']} Hz`\n"
           f"━━━━━━━━━━━━━━━━━━━━")

    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
        
        for fig, title in [(fig_timeline, "Flux Harmonique"), (fig_radar, "Signature Spectrale")]:
            img_bytes = fig.to_image(format="png", engine="kaleido")
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", 
                          data={"chat_id": CHAT_ID, "caption": f" {title} - {data['name']}"},
                          files={"photo": img_bytes})
    except Exception as e:
        st.error(f"Erreur Telegram: {e}")

# --- FILTRES ---
def apply_2026_filters(y, sr):
    y = librosa.effects.preemphasis(y)
    y_harm, _ = librosa.effects.hpss(y, margin=(10.0, 2.0))
    nyq = 0.5 * sr
    low, high = 100 / nyq, 3000 / nyq
    b, a = butter(6, [low, high], btype='band')
    return lfilter(b, a, y_harm)

# --- FUSION CHROMA DYNAMIQUE (Niveau 2) ---
def compute_entropy(chroma):
    p = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-10)
    return -np.sum(p * np.log(p + 1e-10), axis=0).mean()

def multi_chroma_fusion(y, sr, tuning):
    cqt = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning, bins_per_octave=72, n_octaves=7)
    cens = librosa.feature.chroma_cens(y=y, sr=sr)
    stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=8192)
    
    ent_cqt = compute_entropy(cqt)
    ent_cens = compute_entropy(cens)
    ent_stft = compute_entropy(stft)
    
    ents = np.array([ent_cqt, ent_cens, ent_stft])
    if np.all(ents == 0):
        weights = np.array([0.5, 0.3, 0.2])
    else:
        weights = 1.0 / (ents + 1e-8)
        weights /= weights.sum()
    
    fused = weights[0] * cqt + weights[1] * cens + weights[2] * stft
    return scipy.ndimage.median_filter(fused, size=(1, 15))

# --- MOTEUR DE TRAITEMENT V4 (avec tous les niveaux + tempo robuste) ---
def analyze_engine_v4(file_object, file_name, nb_segments=48):
    try:
        y, sr = librosa.load(file_object, sr=22050)
    except Exception as e:
        st.error(f"Erreur chargement audio {file_name}: {e}")
        return None
    
    tuning = librosa.estimate_tuning(y=y, sr=sr, bins_per_octave=72)
    y_clean = apply_2026_filters(y, sr)
    chroma_fused = multi_chroma_fusion(y_clean, sr, tuning)
    duration = librosa.get_duration(y=y, sr=sr)
    
    steps = np.linspace(0, chroma_fused.shape[1], nb_segments + 1, dtype=int)
    results_stream = []
    templates = generate_real_templates(sr=sr)
    
    for i in range(len(steps)-1):
        segment = chroma_fused[:, steps[i]:steps[i+1]]
        if segment.shape[1] == 0:
            continue
        avg_chroma = np.mean(segment, axis=1)
        avg_chroma_norm = (avg_chroma - np.mean(avg_chroma)) / (np.std(avg_chroma) + 1e-8)
        
        best_score_template = -1
        best_key_template = "Ambiguous"
        
        for key, temp in templates.items():
            score = np.corrcoef(avg_chroma_norm, temp)[0, 1]
            root_idx = NOTES.index(key.split()[0])
            root_strength = avg_chroma[root_idx] / (np.mean(avg_chroma) + 1e-8)
            if root_idx == np.argmax(avg_chroma) and root_strength > 1.5:
                score *= 1.15  # Boost modéré si root très dominante
            
            if score > best_score_template:
                best_score_template = score
                best_key_template = key
        
        sof_key, sof_score = signature_of_fifths_key(avg_chroma)
        
        # Niveau 3 : fusion pondérée des deux méthodes
        conf_template = best_score_template / (np.max(avg_chroma_norm) + 1e-8)
        weight_template = 0.65 if conf_template > 0.75 else 0.40
        combined_score = weight_template * best_score_template + (1 - weight_template) * sof_score
        
        if weight_template > 0.5:
            best_key = best_key_template
            best_score = combined_score
        else:
            best_key = sof_key
            best_score = combined_score
        
        time_start = (steps[i] / chroma_fused.shape[1]) * duration
        results_stream.append({"time": time_start, "key": best_key, "score": best_score})
    
    # Niveau 4 : vote pondéré par durée × score + seuil de confiance
    weighted_keys = defaultdict(float)
    prev_time = 0.0
    for r in results_stream:
        seg_duration = r['time'] - prev_time
        weighted_keys[r['key']] += r['score'] * seg_duration
        prev_time = r['time']
    
    if not weighted_keys:
        main_key = "Ambiguous"
    else:
        total_weight = sum(weighted_keys.values())
        main_key = max(weighted_keys, key=weighted_keys.get)
        confidence = weighted_keys[main_key] / total_weight if total_weight > 0 else 0
        
        if confidence < 0.68:
            main_key = f"Ambiguous ({main_key} faible confiance)"
    
    # Tempo robuste : gestion du cas nan
    _, y_perc = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
    tempo_val = float(tempo)
    tempo_display = int(tempo_val) if not np.isnan(tempo_val) else "???"

    return {
        "key": main_key,
        "camelot": CAMELOT_MAP.get(main_key.split(" (")[0], "??"),
        "tempo": tempo_display,
        "tuning": round(440 * (2 ** (tuning / 12)), 1),
        "timeline": results_stream,
        "name": file_name,
        "chroma_avg": np.mean(chroma_fused, axis=1)
    }

# --- INTERFACE ---
st.title("🎧 L'Elite - V4 (Améliorations 2026)")

with st.sidebar:
    st.header("⚙️ Configuration")
    if TELEGRAM_TOKEN and CHAT_ID:
        st.success("Telegram Secret : OK")
    else:
        st.error("Telegram Secret : MANQUANT")
    
    nb_segments = st.slider("Nombre de segments d'analyse", 24, 80, 48)
    
    if st.button("Reset Cache"):
        st.cache_data.clear()
        st.rerun()

files = st.file_uploader("Upload Audio", type=['mp3','wav','flac'], accept_multiple_files=True)

if files:
    for f in reversed(files):
        with st.spinner(f"Analyse Deep Fusion V4 : {f.name}"):
            data = analyze_engine_v4(f, f.name, nb_segments=nb_segments)
            if data is None:
                continue
            
        with st.expander(f"📊 {data['name']}", expanded=True):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                key_display = data['key'].upper() if "Ambiguous" not in data['key'] else data['key']
                st.markdown(f"""
                    <div style="background:#1e293b; padding:20px; border-radius:15px; border-left: 5px solid #3b82f6;">
                        <h2 style="color:#60a5fa; margin:0;">{key_display}</h2>
                        <h1 style="font-size:3em; margin:0; color: white;">{data['camelot']}</h1>
                        <p style="color: white; opacity: 0.9; margin:0;">{data['tempo']} BPM | {data['tuning']} Hz</p>
                    </div>
                """, unsafe_allow_html=True)
                
                fig_polar = go.Figure(data=go.Scatterpolar(r=data['chroma_avg'], theta=NOTES, fill='toself', line_color='#60a5fa'))
                fig_polar.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_polar, use_container_width=True)

            with col2:
                df_timeline = pd.DataFrame(data['timeline'])
                fig_line = px.line(df_timeline, x="time", y="key", title="Stabilité Harmonique (V4)",
                                   markers=True, template="plotly_dark", color_discrete_sequence=["#3b82f6"])
                st.plotly_chart(fig_line, use_container_width=True)

            if TELEGRAM_TOKEN and CHAT_ID:
                send_telegram_expert(data, fig_line, fig_polar)
                st.toast(f"Rapport envoyé pour {data['name']}")
        
        gc.collect()
