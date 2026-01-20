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

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="L'Elite", page_icon="🎵", layout="wide")

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

# ────────────────────────────────────────────────────────────────
# VALIDATION TRIAD (nouveau)
# ────────────────────────────────────────────────────────────────
def validate_triad(chroma_norm, root_idx, mode):
    """
    Vérifie que les trois notes du triad sont suffisamment présentes
    Retourne: (valide: bool, force_triad: float)
    """
    if mode == "major":
        third_offset = 4
    else:
        third_offset = 3
    
    fifth_offset = 7
    
    root_strength   = chroma_norm[root_idx]
    third_strength  = chroma_norm[(root_idx + third_offset) % 12]
    fifth_strength  = chroma_norm[(root_idx + fifth_offset) % 12]
    
    triad_avg = (root_strength + third_strength + fifth_strength) / 3.0
    
    # Seuils raisonnables – à ajuster selon tes tests
    valid = (
        triad_avg     >= 0.35 and
        third_strength >= 0.30 and   # tierce très importante
        fifth_strength >= 0.25
    )
    
    return valid, triad_avg


def get_triad_debug(chroma_norm, key):
    """Retourne une string de debug pour le triad"""
    if " " not in key or key == "Ambiguous":
        return "—"
    root, mode = key.split()
    root_idx = NOTES.index(root)
    third_offset = 4 if mode == "major" else 3
    fifth_offset = 7
    return (
        f"Root: **{root}** ({chroma_norm[root_idx]:.3f})  •  "
        f"Tierce: **{NOTES[(root_idx + third_offset)%12]}** ({chroma_norm[(root_idx + third_offset)%12]:.3f})  •  "
        f"Quinte: **{NOTES[(root_idx + fifth_offset)%12]}** ({chroma_norm[(root_idx + fifth_offset)%12]:.3f})"
    )


# --- GÉNÉRATION DE TEMPLATES ---
@st.cache_resource
def generate_real_templates(sr=22050, A4=440.0, duration=1.0):
    templates = {}
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
            for f in freqs:
                y += np.sin(2 * np.pi * f * t)
                y += 0.4 * np.sin(2 * np.pi * 2 * f * t)
                y += 0.2 * np.sin(2 * np.pi * 3 * f * t)
            
            y = librosa.util.normalize(y)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
            chroma_avg = np.mean(chroma, axis=1)
            chroma_avg = (chroma_avg - np.mean(chroma_avg)) / (np.std(chroma_avg) + 1e-8)
            templates[f"{root} {mode}"] = chroma_avg
    return templates


def signature_of_fifths_key(chroma_avg):
    fifths_order = [0,7,2,9,4,11,6,1,8,3,10,5]
    weights = [1, 0.9, 0.75, 0.6, 0.45, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04]
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


# --- FILTRES & EXTRACTION CHROMA ---
def apply_2026_filters(y, sr):
    y = librosa.effects.preemphasis(y)
    y_harm, _ = librosa.effects.hpss(y, margin=(10.0, 2.0))
    nyq = 0.5 * sr
    low, high = 100 / nyq, 3000 / nyq
    b, a = butter(6, [low, high], btype='band')
    return lfilter(b, a, y_harm)


def multi_chroma_fusion(y, sr, tuning):
    cqt = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning, bins_per_octave=72, n_octaves=7)
    cens = librosa.feature.chroma_cens(y=y, sr=sr)
    stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=8192)
    fused = (0.5 * cqt) + (0.3 * cens) + (0.2 * stft)
    return scipy.ndimage.median_filter(fused, size=(1, 15))


# ────────────────────────────────────────────────────────────────
# MOTEUR D'ANALYSE – VERSION AVEC VALIDATION TRIAD
# ────────────────────────────────────────────────────────────────
def analyze_engine_v3(file_object, file_name):
    y, sr = librosa.load(file_object, sr=22050)
    
    tuning = librosa.estimate_tuning(y=y, sr=sr, bins_per_octave=72)
    y_clean = apply_2026_filters(y, sr)
    chroma_fused = multi_chroma_fusion(y_clean, sr, tuning)
    duration = librosa.get_duration(y=y, sr=sr)
    
    steps = np.linspace(0, chroma_fused.shape[1], 40, dtype=int)
    results_stream = []
    templates = generate_real_templates(sr=sr)
    
    for i in range(len(steps)-1):
        segment = chroma_fused[:, steps[i]:steps[i+1]]
        avg_chroma = np.mean(segment, axis=1)
        avg_chroma_norm = (avg_chroma - np.mean(avg_chroma)) / (np.std(avg_chroma) + 1e-8)
        
        best_score = -1
        best_key = "Ambiguous"
        best_triad_ok = False
        best_triad_strength = 0.0
        
        # 1. Recherche parmi les templates → avec filtre triad
        for key, temp in templates.items():
            score = np.corrcoef(avg_chroma_norm, temp)[0, 1]
            root_name = key.split()[0]
            root_idx = NOTES.index(root_name)
            mode = key.split()[1]
            
            if np.argmax(avg_chroma_norm) == root_idx:
                score *= 1.2
            
            triad_valid, triad_strength = validate_triad(avg_chroma_norm, root_idx, mode)
            
            if triad_valid and score > best_score:
                best_score = score
                best_key = key
                best_triad_ok = True
                best_triad_strength = triad_strength
        
        # 2. Aucun template n'a validé le triad → fallback circle of fifths
        if not best_triad_ok:
            sof_key, sof_score = signature_of_fifths_key(avg_chroma_norm)
            best_key = sof_key
            # Tu peux ajouter " ?" ou diminuer la confiance si tu veux :
            # best_key = sof_key + " ?"
        
        results_stream.append({
            "time": (steps[i] / chroma_fused.shape[1]) * duration,
            "key": best_key,
            "score": best_score if best_triad_ok else None,
            "triad_ok": best_triad_ok,
            "triad_strength": best_triad_strength
        })

    # Tonalité principale (la plus fréquente, en ignorant Ambiguous si possible)
    keys_found = [r['key'] for r in results_stream if r['key'] != "Ambiguous"]
    main_key = Counter(keys_found).most_common(1)[0][0] if keys_found else "Ambiguous"

    # Tempo
    _, y_perc = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)

    return {
        "key": main_key,
        "camelot": CAMELOT_MAP.get(main_key, "??"),
        "tempo": int(float(tempo or 0)),
        "tuning": round(440 * (2 ** (tuning / 12)), 1),
        "timeline": results_stream,
        "name": file_name,
        "chroma_avg": np.mean(chroma_fused, axis=1)
    }


# ────────────────────────────────────────────────────────────────
# INTERFACE UTILISATEUR
# ────────────────────────────────────────────────────────────────
st.title("🎵 L'Elite – Analyse Tonalité & Camelot")

with st.sidebar:
    st.header("Configuration")
    if TELEGRAM_TOKEN and CHAT_ID:
        st.success("Telegram configuré ✓")
    else:
        st.error("Secrets Telegram manquants")
    
    if st.button("Reset Cache & Rerun"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

files = st.file_uploader("Déposer fichiers audio", type=['mp3','wav','flac'], accept_multiple_files=True)

if files:
    for f in reversed(files):
        with st.spinner(f"Analyse → {f.name}"):
            data = analyze_engine_v3(f, f.name)
        
        with st.expander(f"📁 {data['name']}", expanded=True):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                    <div style="background:#1e293b; padding:24px; border-radius:16px; border-left:6px solid #3b82f6;">
                        <h2 style="color:#60a5fa; margin:0 0 8px 0;">{data['key'].upper()}</h2>
                        <h1 style="font-size:3.8em; margin:0; color:white;">{data['camelot']}</h1>
                        <p style="color:#cbd5e1; margin:12px 0 0 0; font-size:1.1em;">
                            {data['tempo']} BPM  •  {data['tuning']} Hz
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                fig_polar = go.Figure(data=go.Scatterpolar(
                    r=data['chroma_avg'],
                    theta=NOTES,
                    fill='toself',
                    line_color='#60a5fa'
                ))
                fig_polar.update_layout(
                    template="plotly_dark",
                    height=320,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_polar, use_container_width=True)
                
                # Debug triad (optionnel – tu peux commenter si trop verbeux)
                with st.expander("🔍 Debug triad", expanded=False):
                    st.markdown(get_triad_debug(data['chroma_avg'], data['key']))
            
            with col2:
                df_timeline = pd.DataFrame(data['timeline'])
                fig_line = px.line(
                    df_timeline,
                    x="time",
                    y="key",
                    title="Évolution harmonique",
                    markers=True,
                    template="plotly_dark",
                    color_discrete_sequence=["#3b82f6"]
                )
                fig_line.update_traces(line=dict(width=2.2))
                st.plotly_chart(fig_line, use_container_width=True)

            if TELEGRAM_TOKEN and CHAT_ID:
                send_telegram_expert(data, fig_line, fig_polar)
                st.toast(f"Rapport envoyé pour {data['name']}", icon="📤")
        
        gc.collect()
