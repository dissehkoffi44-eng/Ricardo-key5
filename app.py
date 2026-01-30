# RCDJ228 SNIPER M3 - VERSION FUSIONNÉE PRO (MOTEUR PIANO COMPANION)
# LOGIQUE : Corrélation Bellman + Masques Théoriques Piano Companion + 7 Modes Grecs
# FIX : Sécurité sur longueur de segment pour éviter ParameterError

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

# --- FORCE FFMPEG PATH (WINDOWS FIX) ---
if os.path.exists(r'C:\ffmpeg\bin'):
    os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 MUSIC SNIPER", page_icon="🎯", layout="wide")

TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES & LOGIQUE PIANO COMPANION ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'locrian']]

# Masques théoriques Piano Companion (1=autorisé, 0=interdit)
COMPANION_MASKS = {
    "major":     [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], 
    "minor":     [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], 
    "dorian":    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    "phrygian":  [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    "lydian":    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    "mixolydian":[1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    "locrian":   [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
}

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

PROFILES = {
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4]
    }
}

# --- FONCTIONS UTILITAIRES ---

def get_neighbor_camelot(camelot_str: str, offset: int) -> str:
    if camelot_str in ['??', None, '']: return '??'
    try:
        num = int(camelot_str[:-1])
        wheel = camelot_str[-1]
        new_num = ((num - 1 + offset) % 12) + 1
        return f"{new_num}{wheel}"
    except: return '??'

def get_theoretical_score(chroma_vector, key_name):
    root_note, mode = key_name.split(maxsplit=1)
    root_idx = NOTES_LIST.index(root_note)
    mask = np.roll(COMPANION_MASKS.get(mode, COMPANION_MASKS["major"]), root_idx)
    total_energy = np.sum(chroma_vector) + 1e-6
    allowed_energy = np.sum(chroma_vector * mask)
    pilar_penalty = 1.0
    if chroma_vector[root_idx] < 0.35: pilar_penalty *= 0.6
    if chroma_vector[(root_idx + 7) % 12] < 0.25: pilar_penalty *= 0.8
    return (allowed_energy / total_energy) * pilar_penalty

def get_mixing_advice(data):
    if not data.get('modulation', False): return None
    principal_camelot = data.get('camelot', '??')
    target_key = data.get('target_key', 'Inconnu')
    target_camelot = data.get('target_camelot', '??')
    perc = data.get('mod_target_percentage', 0)
    ends_in_target = data.get('mod_ends_in_target', False)
    time_str = data.get('modulation_time_str', '??:??')

    lines = ["**Checklist mix harmonique (Logique Piano Companion) :**"]
    if ends_in_target:
        lines.append(f"✅ **Le morceau termine dans {target_key.upper()} ({target_camelot})**")
        priority = "target"
    else:
        lines.append(f"⚠️ **Ne termine pas en {target_key.upper()}**. Sortie : {principal_camelot}")
        priority = "principal"

    lines.append(f"⚠️ **Moment de bascule ≈ {time_str}**")
    lines.append(f"\n**Choix safe :** {target_camelot if priority=='target' else principal_camelot} (ou ±1)")
    lines.append(f"**Montée Punchy (+3) :** {get_neighbor_camelot(target_camelot if priority=='target' else principal_camelot, 3)}")
    
    return "\n".join(lines)

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-card { padding: 40px; border-radius: 30px; text-align: center; color: white; border: 1px solid rgba(99, 102, 241, 0.3); box-shadow: 0 15px 45px rgba(0,0,0,0.6); margin-bottom: 20px; }
    .file-header { background: #1f2937; color: #10b981; padding: 10px 20px; border-radius: 10px; font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 10px; border-left: 5px solid #10b981; }
    .modulation-alert { background: rgba(239, 68, 68, 0.20); color: #fca5a5; padding: 18px; border-radius: 12px; border: 1px solid #ef4444; margin: 20px 0; font-weight: bold; line-height: 1.6; }
    .metric-box { background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d; height: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEUR DE CALCUL ---

def apply_sniper_filters(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    nyq = 0.5 * sr
    b, a = butter(4, [80/nyq, 5000/nyq], btype='band')
    return lfilter(b, a, y_harm)

def solve_key_sniper(chroma_vector):
    best_overall_score = -1
    best_key = "Unknown"
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    
    for mode in COMPANION_MASKS.keys():
        profile = PROFILES["bellman"]["major" if mode != "minor" else "minor"]
        for i in range(12):
            stat_score = np.corrcoef(cv, np.roll(profile, i))[0, 1]
            key_name = f"{NOTES_LIST[i]} {mode}"
            theo_fit = get_theoretical_score(cv, key_name)
            final_score = stat_score * (theo_fit ** 1.8) # Puissance théorique renforcée
            
            if final_score > best_overall_score:
                best_overall_score = final_score
                best_key = key_name
    return {"key": best_key, "score": best_overall_score}

def seconds_to_mmss(seconds):
    if seconds is None: return "??:??"
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def process_audio_precision(file_bytes, file_name, _progress_callback=None):
    try:
        if file_name.lower().endswith('m4a'):
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            y = (samples.reshape((-1, 2)).mean(axis=1) if audio.channels == 2 else samples) / (2**15)
            sr = audio.frame_rate
            if sr != 22050: y = librosa.resample(y, orig_sr=sr, target_sr=22050); sr = 22050
        else:
            with io.BytesIO(file_bytes) as buf: y, sr = librosa.load(buf, sr=22050, mono=True)
    except Exception as e:
        st.error(f"Erreur: {e}"); return None

    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_sniper_filters(y, sr)

    step, timeline, votes = 6, [], Counter()
    segments = list(range(0, max(1, int(duration) - step), 3))
    
    for idx, start in enumerate(segments):
        if _progress_callback: _progress_callback(int((idx/len(segments))*100), f"Scan {start}s / {int(duration)}s")
        idx_s, idx_e = int(start * sr), int((start + step) * sr)
        seg = y_filt[idx_s:idx_e]
        
        # FIX : Sécurité taille minimum pour Chroma CQT (8192 samples minimum conseillés)
        if len(seg) < 8192 or np.max(np.abs(seg)) < 0.015: continue
        
        c_raw = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=24)
        c_avg = np.mean((c_raw[::2, :] + c_raw[1::2, :]) / 2, axis=1)
        
        res = solve_key_sniper(c_avg)
        weight = 2.0 if (start < 15 or start > (duration - 20)) else 1.0
        votes[res['key']] += int(res['score'] * 100 * weight)
        timeline.append({"Temps": start, "Note": res['key'], "Conf": res['score']})

    if not votes: return None

    most_common = votes.most_common(2)
    final_key = most_common[0][0]
    final_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == final_key]) * 100)
    
    mod_detected = len(most_common) > 1 and (votes[most_common[1][0]] / sum(votes.values())) > 0.22
    target_key = most_common[1][0] if mod_detected else None

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma_avg = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)

    res_obj = {
        "key": final_key, "camelot": CAMELOT_MAP.get(final_key, "??"),
        "conf": min(final_conf, 99), "tempo": int(float(tempo)),
        "tuning": round(440 * (2**(tuning/12)), 1), "timeline": timeline,
        "chroma": chroma_avg.tolist(), "modulation": mod_detected,
        "target_key": target_key, "target_camelot": CAMELOT_MAP.get(target_key, "??") if target_key else None,
        "mod_target_percentage": round((sum(1 for t in timeline if t["Note"] == target_key)/len(timeline))*100, 1) if mod_detected else 0,
        "mod_ends_in_target": (timeline[-1]["Note"] == target_key) if mod_detected else False,
        "modulation_time_str": seconds_to_mmss(min([t["Temps"] for t in timeline if t["Note"] == target_key])) if mod_detected else None,
        "name": file_name
    }

    # ENVOI TELEGRAM COMPLET
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            caption = (f"🎯 *RCDJ228 SNIPER PRO REPORT*\n━━━━━━━━━━━━\n"
                       f"Track: `{file_name}`\nKey: `{final_key.upper()}` ({res_obj['camelot']})\n"
                       f"Conf: `{res_obj['conf']}%` | Tempo: `{res_obj['tempo']} BPM`\n"
                       f"{'⚠️ MODULATION: ' + target_key if mod_detected else '✅ STABLE'}")
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={'chat_id': CHAT_ID, 'text': caption, 'parse_mode': 'Markdown'})
        except: pass

    return res_obj

def get_chord_js(btn_id, key_str):
    parts = key_str.split()
    note = parts[0]
    mode = parts[1] if len(parts) > 1 else 'major'
    return f"""
    document.getElementById('{btn_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
        const intervals = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach(i => {{
            const o = ctx.createOscillator(); const g = ctx.createGain();
            o.type = 'triangle'; o.frequency.setValueAtTime(freqs['{note}'] * Math.pow(2, i/12), ctx.currentTime);
            g.gain.setValueAtTime(0, ctx.currentTime); g.gain.linearRampToValueAtTime(0.1, ctx.currentTime + 0.1);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.5);
            o.connect(g); g.connect(ctx.destination); o.start(); o.stop(ctx.currentTime + 1.5);
        }});
    }}; """

# --- INTERFACE PRINCIPALE ---

st.title("🎯 RCDJ228 MUSIC SNIPER PRO")

uploaded_files = st.file_uploader("Audio files", type=['mp3','wav','flac','m4a'], accept_multiple_files=True)

if uploaded_files:
    container = st.container()
    for i, f in enumerate(reversed(uploaded_files)):
        with st.status(f"Sniper Pro : {f.name}") as status:
            data = process_audio_precision(f.getvalue(), f.name, lambda v, m: status.update(label=m))
            status.update(label="Certifié par Piano Companion Engine", state="complete")

        if data:
            with container:
                st.markdown(f"<div class='file-header'>{data['name']}</div>", unsafe_allow_html=True)
                
                # Report Card
                st.markdown(f"""<div class="report-card" style="background:linear-gradient(135deg, #064e3b, #0b0e14);">
                    <h1 style="font-size:5.5em; margin:0;">{data['key'].upper()}</h1>
                    <p style="font-size:1.5em; opacity:0.9;">CAMELOT {data['camelot']} • CONF {data['conf']}%</p>
                </div>""", unsafe_allow_html=True)

                if data['modulation']:
                    st.markdown(f"""<div class="modulation-alert">⚠️ MODULATION : {data['target_key'].upper()} ({data['target_camelot']}) à {data['modulation_time_str']} ({data['mod_target_percentage']}%)</div>""", unsafe_allow_html=True)

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.markdown(f"<div class='metric-box'>TEMPO<br><span style='font-size:2.4em; color:#10b981;'>{data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-box'>TUNING<br><span style='font-size:2.2em; color:#58a6ff;'>{data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)
                with m3:
                    btn_id = f"play_{i}"
                    components.html(f"<button id='{btn_id}' style='width:100%; height:90px; background:#4F46E5; color:white; border:none; border-radius:15px; font-weight:bold; cursor:pointer;'>TESTER L'ACCORD</button><script>{get_chord_js(btn_id, data['key'])}</script>", height=100)

                # Advice
                advice = get_mixing_advice(data)
                if advice:
                    with st.expander("📋 Checklist Mix (Piano Companion Logic)", expanded=True):
                        st.markdown(f"<div style='background:rgba(16,185,129,0.1); padding:15px; border-radius:10px;'>{advice}</div>", unsafe_allow_html=True)

                # Charts
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_tl = px.line(pd.DataFrame(data['timeline']), x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER})
                    st.plotly_chart(fig_tl, use_container_width=True)
                with c2:
                    fig_rd = go.Figure(data=go.Scatterpolar(r=data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
                    fig_rd.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
                    st.plotly_chart(fig_rd, use_container_width=True)
                st.divider()

with st.sidebar:
    st.header("Sniper Control")
    if st.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.rerun()
