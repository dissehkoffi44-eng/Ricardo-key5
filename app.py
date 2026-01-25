# RCDJ228 MUSIC SNIPER M4 - VERSION FUSION OPTIMISÉE (2026)
# Précision Code 2 + Esthétique & UX Code 1

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

# Force FFMPEG path (Windows fix)
if os.path.exists(r'C:\ffmpeg\bin'):
    os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'

# ────────────────────────────────────────────────
# CONFIGURATION PAGE
# ────────────────────────────────────────────────
st.set_page_config(page_title="RCDJ228 MUSIC SNIPER M4", page_icon="🔫🎵", layout="wide")

TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID        = st.secrets.get("CHAT_ID")

# ────────────────────────────────────────────────
# CONSTANTES & RÉFÉRENTIELS
# ────────────────────────────────────────────────
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

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

WEIGHTS = {
    "global":    0.60,
    "segments":  0.30,
    "bass_bonus": 0.10
}

# ────────────────────────────────────────────────
# STYLES CSS
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-card { 
        padding: 40px; border-radius: 30px; text-align: center; color: white; 
        border: 1px solid rgba(99, 102, 241, 0.3); box-shadow: 0 15px 45px rgba(0,0,0,0.6);
        margin-bottom: 20px;
    }
    .file-header {
        background: #1f2937; color: #10b981; padding: 10px 20px; border-radius: 10px;
        font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 10px;
        border-left: 5px solid #10b981;
    }
    .modulation-alert {
        background: rgba(239, 68, 68, 0.15); color: #f87171;
        padding: 15px; border-radius: 15px; border: 1px solid #ef4444;
        margin-top: 20px; font-weight: bold; font-family: 'JetBrains Mono', monospace;
    }
    .metric-box {
        background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d;
        height: 100%; transition: 0.3s;
    }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# FONCTIONS TECHNIQUES
# ────────────────────────────────────────────────

def butter_lowpass(y, sr, cutoff=150):
    nyq = 0.5 * sr
    b, a = butter(4, cutoff / nyq, btype='low')
    return lfilter(b, a, y)

def apply_precision_filters(y, sr):
    y_harm, _ = librosa.effects.hpss(y, margin=(1.2, 4.5))
    nyq = 0.5 * sr
    b, a = butter(4, [60/nyq, 5000/nyq], btype='band')
    return lfilter(b, a, y_harm)

def vote_profiles(chroma_cqt, chroma_cens, bass_chroma):
    cv = (chroma_cqt - chroma_cqt.min()) / (chroma_cqt.max() - chroma_cqt.min() + 1e-8)
    cens_norm = (chroma_cens - chroma_cens.min()) / (chroma_cens.max() - chroma_cens.min() + 1e-8)
    bv = (bass_chroma - bass_chroma.min()) / (bass_chroma.max() - bass_chroma.min() + 1e-8)

    scores = {f"{n} {m}": 0.0 for n in NOTES_LIST for m in ["major", "minor"]}

    for profile in PROFILES.values():
        for mode in ["major", "minor"]:
            for i in range(12):
                corr_cqt  = np.corrcoef(cv,   np.roll(profile[mode], i))[0,1]
                corr_cens = np.corrcoef(cens_norm, np.roll(profile[mode], i))[0,1]
                combined  = 0.70 * corr_cqt + 0.30 * corr_cens

                bonus = (
                    bv[i]                * 0.40 +   # racine basse
                    cv[(i+7)%12]         * 0.18 +   # quinte
                    (cv[i] + bv[i])/2   * 0.12     # renfort racine
                )
                scores[f"{NOTES_LIST[i]} {mode}"] += (combined + bonus) / len(PROFILES)
    return scores

def process_audio_precision(file_bytes, file_name, progress_callback=None):
    ext = file_name.split('.')[-1].lower()
    sr_target = 22050

    try:
        if ext == 'm4a':
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            y = samples / (2 ** (8 * audio.sample_width - 1))
            sr = audio.frame_rate
            if sr != sr_target:
                y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
                sr = sr_target
        else:
            with io.BytesIO(file_bytes) as buf:
                y, sr = librosa.load(buf, sr=sr_target, mono=True)
    except Exception as e:
        st.error(f"Erreur lecture {file_name}: {e}")
        return None

    duration = librosa.get_duration(y=y, sr=sr)
    tuning_offset = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_precision_filters(y, sr)

    # ── Analyse globale
    chroma_cqt_glob  = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning_offset), axis=1)
    chroma_cens_glob = np.mean(librosa.feature.chroma_cens(y=y_filt, sr=sr, tuning=tuning_offset), axis=1)
    bass_glob        = np.mean(librosa.feature.chroma_cqt(y=butter_lowpass(y, sr), sr=sr), axis=1)
    global_scores    = vote_profiles(chroma_cqt_glob, chroma_cens_glob, bass_glob)

    # ── Analyse segments
    seg_duration, overlap = 12, 6
    step = seg_duration - overlap
    segment_votes = Counter()
    timeline = []
    valid_segments = 0

    segments_starts = list(range(0, max(1, int(duration) - seg_duration), step))

    for idx, start_s in enumerate(segments_starts):
        if progress_callback:
            prog = int((idx + 1) / len(segments_starts) * 100)
            progress_callback(prog, f"Segment {idx+1}/{len(segments_starts)} — {start_s:.1f}s")

        seg_start_idx = int(start_s * sr)
        seg_end_idx   = int((start_s + seg_duration) * sr)
        y_seg = y_filt[seg_start_idx:seg_end_idx]

        if len(y_seg) < 1000 or np.max(np.abs(y_seg)) < 0.015:
            continue

        cqt_seg  = np.mean(librosa.feature.chroma_cqt(y=y_seg, sr=sr, tuning=tuning_offset), axis=1)
        cens_seg = np.mean(librosa.feature.chroma_cens(y=y_seg, sr=sr, tuning=tuning_offset), axis=1)
        bass_seg = np.mean(librosa.feature.chroma_cqt(y=butter_lowpass(y_seg, sr), sr=sr), axis=1)

        seg_scores = vote_profiles(cqt_seg, cens_seg, bass_seg)
        best_key = max(seg_scores, key=seg_scores.get)

        if seg_scores[best_key] >= 0.68:
            weight = 1.40 if 0.20 < (start_s / duration) < 0.80 else 1.0
            segment_votes[best_key] += seg_scores[best_key] * weight
            timeline.append({"time": start_s + seg_duration/2, "key": best_key, "score": seg_scores[best_key]})
            valid_segments += 1

    if not segment_votes and not global_scores:
        return None

    # ── Score final pondéré
    total_seg = sum(segment_votes.values()) or 1
    seg_norm = {k: v / total_seg for k,v in segment_votes.items()}

    final_scores = Counter()
    for k in set(global_scores) | set(seg_norm):
        final_scores[k] = (global_scores.get(k, 0) * WEIGHTS["global"] + seg_norm.get(k, 0) * WEIGHTS["segments"])

    best_key, best_raw_score = final_scores.most_common(1)[0]
    max_possible = max(final_scores.values()) if final_scores else 1
    confidence = min(99, int(100 * best_raw_score / max_possible * 1.15))

    # Modulation
    modulation = None
    if len(timeline) >= 6:
        mid = len(timeline) // 2
        first_half = Counter([t["key"] for t in timeline[:mid]])
        second_half = Counter([t["key"] for t in timeline[mid:]])
        if first_half and second_half:
            top1 = first_half.most_common(1)[0][0]
            top2 = second_half.most_common(1)[0][0]
            if top1 != top2 and second_half[top2] > first_half[top2] * 1.4:
                modulation = top2

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    result = {
        "name": file_name,
        "key": best_key,
        "camelot": CAMELOT_MAP.get(best_key, "??"),
        "conf": confidence,
        "tempo": int(round(float(tempo))),
        "tuning_hz": round(440 * (2 ** (tuning_offset / 12)), 1),
        "tuning_cents": round(tuning_offset * 100, 1),
        "modulation": modulation,
        "target_camelot": CAMELOT_MAP.get(modulation, "??") if modulation else None,
        "timeline": timeline,
        "chroma": chroma_cqt_glob.tolist(),
        "valid_segments": valid_segments,
        "duration": round(duration, 1)
    }

    # ── Envoi Telegram
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            df_tl = pd.DataFrame(timeline)
            fig_tl = px.line(df_tl, x="time", y="key", markers=True, template="plotly_dark", category_orders={"key": NOTES_ORDER})
            fig_tl.update_layout(height=450, margin=dict(l=20,r=20,t=30,b=20))
            img_tl = fig_tl.to_image(format="png", width=1000, height=500)

            fig_rd = go.Figure(data=go.Scatterpolar(r=result["chroma"], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
            fig_rd.update_layout(template="plotly_dark", height=500, polar=dict(radialaxis=dict(visible=False)), margin=dict(l=40,r=40,t=30,b=30))
            img_rd = fig_rd.to_image(format="png", width=600, height=600)

            mod_status = f"**MODULATION →** {modulation.upper()} ({result['target_camelot']})" if modulation else "**STABLE**"
            caption = (
                f"**RCDJ228 SNIPER M4 RAPPORT**\n━━━━━━━━━━━━━━\n"
                f"**Fichier** `{file_name}`\n"
                f"**Tonalité** `{best_key.upper()}`\n"
                f"**Camelot** `{result['camelot']}`\n"
                f"**Confiance** `{confidence}%`\n"
                f"**Tempo** `{result['tempo']} BPM`\n"
                f"**Accordage** `{result['tuning_hz']} Hz ({result['tuning_cents']:+.1f}¢)`\n"
                f"**Segments valides** `{valid_segments}`\n"
                f"{mod_status}\n"
                f"━━━━━━━━━━━━━━"
            )

            files = {'p1': ('timeline.png', img_tl, 'image/png'), 'p2': ('radar.png', img_rd, 'image/png')}
            media = [
                {'type': 'photo', 'media': 'attach://p1', 'caption': caption, 'parse_mode': 'Markdown'},
                {'type': 'photo', 'media': 'attach://p2'}
            ]
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMediaGroup", data={'chat_id': CHAT_ID, 'media': json.dumps(media)}, files=files, timeout=20)
        except:
            pass

    del y, y_filt
    gc.collect()
    return result

def get_chord_test_js(btn_id, key_str):
    note, mode = key_str.split()
    return f"""
    document.getElementById('{btn_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const base = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}}['{note}'];
        const intervals = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach(i => {{
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(base * Math.pow(2, i/12), ctx.currentTime);
            gain.gain.setValueAtTime(0, ctx.currentTime);
            gain.gain.linearRampToValueAtTime(0.14, ctx.currentTime + 0.08);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 2.2);
            osc.connect(gain); gain.connect(ctx.destination);
            osc.start(); osc.stop(ctx.currentTime + 2.2);
        }});
    }};
    """

# ────────────────────────────────────────────────
# INTERFACE PRINCIPALE
# ────────────────────────────────────────────────
st.title("🔫 RCDJ228 MUSIC SNIPER M4 — Précision Maximale")

uploaded_files = st.file_uploader("Déposez vos tracks (mp3, wav, flac, m4a)", type=['mp3','wav','flac','m4a'], accept_multiple_files=True)

if uploaded_files:
    total = len(uploaded_files)
    progress_global = st.progress(0)
    status_global = st.empty()
    results_container = st.container()

    for idx, file in enumerate(uploaded_files):
        percent = idx / total
        progress_global.progress(percent)
        status_global.markdown(f"**Analyse {idx+1}/{total}** — {file.name}")

        with st.status(f"Scan → {file.name}", expanded=True) as status:
            inner_prog = st.progress(0)
            inner_text = st.empty()
            def upd_prog(p, msg):
                inner_prog.progress(p/100)
                inner_text.code(msg)
            data = process_audio_precision(file.getvalue(), file.name, upd_prog)
            status.update(label=f"Terminé — {file.name}", state="complete", expanded=False)

        if data:
            with results_container:
                st.markdown(f"<div class='file-header'>RAPPORT → {data['name']}</div>", unsafe_allow_html=True)
                color_bg = "linear-gradient(135deg, #065f46, #064e3b)" if data['conf'] > 88 else "linear-gradient(135deg, #1e293b, #0f172a)"
                st.markdown(f"""
                <div class="report-card" style="background:{color_bg};">
                    <h1 style="font-size:6em; margin:0; font-weight:900;">{data['key'].upper()}</h1>
                    <p style="font-size:1.8em; margin:12px 0;">CAMELOT <b>{data['camelot']}</b>  •  CONFIANCE <b>{data['conf']}%</b></p>
                    {f"<div class='modulation-alert'>MODULATION DÉTECTÉE → {data['modulation'].upper()} ({data['target_camelot']})</div>" if data['modulation'] else ""}
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2.4em;color:#10b981;'>{data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br><span style='font-size:2.4em;color:#58a6ff;'>{data['tuning_hz']}</span><br>Hz ({data['tuning_cents']:+.1f}¢)</div>", unsafe_allow_html=True)
                with col3:
                    btn_id = f"chord_{idx}_{hash(file.name)}"
                    components.html(f"""
                    <button id="{btn_id}" style="width:100%; height:100px; background:linear-gradient(45deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:16px; font-size:1.2em; cursor:pointer;">TESTER L'ACCORD</button>
                    <script>{get_chord_test_js(btn_id, data['key'])}</script>
                    """, height=120)

                c_plot1, c_plot2 = st.columns([2.2, 1])
                with c_plot1:
                    if data["timeline"]:
                        df = pd.DataFrame(data["timeline"])
                        fig = px.line(df, x="time", y="key", markers=True, template="plotly_dark", category_orders={"key": NOTES_ORDER})
                        fig.update_layout(height=340, margin=dict(l=10,r=10,t=10,b=10))
                        st.plotly_chart(fig, use_container_width=True)
                with c_plot2:
                    fig_radar = go.Figure(go.Scatterpolar(r=data["chroma"], theta=NOTES_LIST, fill='toself', line_color='#22c55e'))
                    fig_radar.update_layout(template="plotly_dark", height=340, polar=dict(radialaxis=dict(visible=False)), margin=dict(l=20,r=20,t=10,b=10))
                    st.plotly_chart(fig_radar, use_container_width=True)
                st.markdown("---")

    progress_global.progress(1.0)
    status_global.success(f"**Mission terminée — {total} track(s) analysée(s)**")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2569/2569107.png", width=90)
    st.header("Sniper M4")
    st.caption("Précision 2026 + UX premium")
    if st.button("🔄 Reset cache & relancer"):
        st.cache_data.clear()
        st.rerun()
