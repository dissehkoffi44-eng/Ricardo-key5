import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from datetime import datetime
import io
import streamlit.components.v1 as components
from concurrent.futures import ThreadPoolExecutor

# --- IMPORT POUR LES TAGS MP3 (MUTAGEN) ---
try:
    from mutagen.id3 import ID3, TKEY
    from mutagen.mp3 import MP3
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | V5 Ultra Pro", page_icon="🎧", layout="wide")

# Initialisation des états
if 'history' not in st.session_state:
    st.session_state.history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}
if 'order_list' not in st.session_state:
    st.session_state.order_list = []

# --- DESIGN CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .metric-container { background: white; padding: 20px; border-radius: 15px; border: 1px solid #E0E0E0; text-align: center; height: 100%; transition: transform 0.3s; }
    .metric-container:hover { transform: translateY(-5px); border-color: #6366F1; }
    .label-custom { color: #666; font-size: 0.9em; font-weight: bold; margin-bottom: 5px; }
    .value-custom { font-size: 1.6em; font-weight: 800; color: #1A1A1A; }
    .diag-box { text-align:center; padding:10px; border-radius:10px; border:1px solid #EEE; background: white; }
    /* Extension visuelle de la zone d'upload */
    .stFileUploader {
        border: 2px dashed #6366F1;
        padding: 20px;
        border-radius: 15px;
        background: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEUR AUDIO JS ---
def get_sine_witness(note_mode_str, key_suffix=""):
    parts = note_mode_str.split(' ')
    note = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else "major"
    unique_id = f"playBtn_{note}_{mode}_{key_suffix}".replace("#", "sharp").replace(".", "_")
    
    return components.html(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; font-family: sans-serif;">
        <button id="{unique_id}" style="background: #6366F1; color: white; border: none; border-radius: 50%; width: 28px; height: 28px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 12px;">▶</button>
        <span style="font-size: 9px; font-weight: bold; color: #666;">{note} {mode[:3].upper()}</span>
    </div>
    <script>
    const notesFreq = {{'C':261.63,'C#':277.18,'D':293.66,'D#':311.13,'E':329.63,'F':349.23,'F#':369.99,'G':392.00,'G#':415.30,'A':440.00,'A#':466.16,'B':493.88}};
    const semitones = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
    let audioCtx = null; let oscillators = []; let gainNode = null;
    document.getElementById('{unique_id}').onclick = function() {{
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (this.innerText === '▶') {{
            this.innerText = '◼'; this.style.background = '#E74C3C';
            gainNode = audioCtx.createGain(); gainNode.gain.setValueAtTime(0.05, audioCtx.currentTime);
            gainNode.connect(audioCtx.destination);
            const rootIdx = semitones.indexOf('{note}');
            const intervals = ('{mode}' === 'minor' || '{mode}' === 'dorian') ? [0, 3, 7] : [0, 4, 7];
            intervals.forEach(interval => {{
                let osc = audioCtx.createOscillator(); osc.type = 'sine';
                let freq = notesFreq['{note}'] * Math.pow(2, interval / 12);
                if (!freq) freq = 440; 
                osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
                osc.connect(gainNode); osc.start(); oscillators.push(osc);
            }});
        }} else {{
            oscillators.forEach(o => o.stop()); oscillators = [];
            this.innerText = '▶'; this.style.background = '#6366F1';
        }}
    }};
    </script>
    """, height=40)

# --- MAPPING CAMELOT ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        if mode in ['minor', 'dorian']: return BASE_CAMELOT_MINOR.get(key, "??")
        else: return BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

def get_tagged_audio(file_buffer, key_val):
    if not MUTAGEN_AVAILABLE: return file_buffer
    try:
        file_buffer.seek(0)
        audio_data = io.BytesIO(file_buffer.read())
        audio = MP3(audio_data)
        if audio.tags is None: audio.add_tags()
        audio.tags.add(TKEY(encoding=3, text=key_val))
        output = io.BytesIO()
        audio.save(output)
        output.seek(0)
        return output
    except: return file_buffer

# --- MOTEUR ANALYSE ---
def check_drum_alignment(y, sr):
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_max_mean = np.mean(np.max(chroma, axis=0))
    return flatness < 0.045 or chroma_max_mean > 0.75

def analyze_segment(y, sr):
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning, bins_per_octave=24)
    chroma_avg = np.mean(chroma, axis=1)
    
    PROFILES = {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], 
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], 
        "dorian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.98, 3.34, 3.17],
        "complex": [7.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 5.0, 2.0, 3.5, 4.5, 2.5]
    }
    
    best_score, res_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            actual_rolled = np.roll(chroma_avg, -i)
            if mode == "major" and actual_rolled[4] < actual_rolled[3]: score -= 0.05
            if mode == "minor" and actual_rolled[3] < actual_rolled[4]: score -= 0.05
            if score > best_score: 
                best_score, res_key = score, f"{NOTES[i]} {mode}"
    
    if "complex" in res_key: res_key = res_key.replace("complex", "major")
    return res_key, best_score, chroma_avg

@st.cache_data(show_spinner="Analyse Ultra V5...")
def get_full_analysis(file_buffer):
    file_buffer.seek(0)
    y, sr = librosa.load(file_buffer)
    is_aligned = check_drum_alignment(y, sr)
    y_final, filter_applied = (y, False) if is_aligned else (librosa.effects.hpss(y)[0], True)
    
    duration = librosa.get_duration(y=y_final, sr=sr)
    timeline_data, votes, all_chromas = [], [], []
    
    for start_t in range(0, int(duration) - 10, 10):
        y_seg = y_final[int(start_t*sr):int((start_t+10)*sr)]
        key_seg, score_seg, chroma_vec = analyze_segment(y_seg, sr)
        votes.append(key_seg)
        all_chromas.append(chroma_vec)
        timeline_data.append({"Temps": start_t, "Note": key_seg, "Confiance": round(score_seg * 100, 1)})
    
    dominante_vote = Counter(votes).most_common(1)[0][0]
    avg_chroma_global = np.mean(all_chromas, axis=0)
    
    PROFILES_SYNTH = {"major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]}
    best_synth_score, tonique_synth = -1, ""
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for mode, profile in PROFILES_SYNTH.items():
        for i in range(12):
            score = np.corrcoef(avg_chroma_global, np.roll(profile, i))[0, 1]
            if score > best_synth_score: best_synth_score, tonique_synth = score, f"{NOTES[i]} {mode}"

    top_votes = Counter(votes).most_common(2)
    purity = int((top_votes[0][1] / len(votes)) * 100)
    key_shift_detected = True if len(top_votes) > 1 and (top_votes[1][1] / len(votes)) > 0.25 else False
    stability = Counter(votes).most_common(1)[0][1] / len(votes)
    final_conf = int(max(97, min(99, ((stability*0.5)+(best_synth_score*0.5))*100 + 10))) if dominante_vote == tonique_synth else 91
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = int(np.clip(np.mean(librosa.feature.rms(y=y))*35 + (float(tempo)/160), 1, 10))

    return {
        "file_name": getattr(file_buffer, 'name', 'Unknown'),
        "vote": dominante_vote, "synthese": tonique_synth, "confidence": final_conf, "tempo": int(float(tempo)), 
        "energy": energy, "timeline": timeline_data, "purity": purity, 
        "key_shift": key_shift_detected, "secondary": top_votes[1][0] if len(top_votes)>1 else top_votes[0][0],
        "is_filtered": filter_applied, "original_buffer": file_buffer
    }

# --- INTERFACE ---
st.markdown("<h1 style='text-align: center;'>🎧 RICARDO_DJ228 | V5 ULTRA PRO</h1>", unsafe_allow_html=True)

# Zone d'importation globale (visible partout et prioritée)
files = st.file_uploader("📂 DÉPOSEZ VOS TRACKS ICI (OU CLIQUEZ)", type=['mp3', 'wav', 'flac'], accept_multiple_files=True)

tabs = st.tabs(["📁 ANALYSEUR", "🕒 HISTORIQUE"])

with tabs[0]:
    if files:
        files_to_process = []
        for f in files:
            file_id = f"{f.name}_{f.size}"
            if file_id not in st.session_state.processed_files:
                files_to_process.append(f)
        
        if files_to_process:
            with st.spinner(f"Analyse parallèle V5 ({len(files_to_process)} fichiers)..."):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    new_results = list(executor.map(get_full_analysis, files_to_process))
                    for r in new_results:
                        fid = f"{r['file_name']}_{r['original_buffer'].size}"
                        st.session_state.processed_files[fid] = r
                        # On insère l'ID au DEBUT de la liste pour l'ordre décroissant
                        if fid not in st.session_state.order_list:
                            st.session_state.order_list.insert(0, fid)

        # Affichage selon l'ordre inverse (nouveaux en haut)
        for fid in st.session_state.order_list:
            if fid in st.session_state.processed_files:
                res = st.session_state.processed_files[fid]
                file_name = res['file_name']
                file_buffer = res['original_buffer']
                
                with st.expander(f"🎵 {file_name}", expanded=True):
                    cam_final = get_camelot_pro(res['synthese'])
                    entry = {"Date": datetime.now().strftime("%d/%m %H:%M"), "Fichier": file_name, "Note": res['synthese'], "Camelot": cam_final, "BPM": res['tempo']}
                    if not any(h['Fichier'] == file_name for h in st.session_state.history): 
                        st.session_state.history.insert(0, entry)

                    st.audio(file_buffer) 
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: 
                        st.markdown(f'<div class="metric-container"><div class="label-custom">DOMINANTE</div><div class="value-custom">{res["vote"]}</div><div>{get_camelot_pro(res["vote"])}</div></div>', unsafe_allow_html=True)
                        get_sine_witness(res["vote"], f"dom_{fid}")
                    with c2: 
                        st.markdown(f'<div class="metric-container" style="border-bottom: 4px solid #6366F1;"><div class="label-custom">SYNTHÈSE</div><div class="value-custom">{res["synthese"]}</div><div>{cam_final}</div></div>', unsafe_allow_html=True)
                        get_sine_witness(res["synthese"], f"synth_{fid}")
                        st.download_button(label="💾 MP3 TAGGÉ", data=get_tagged_audio(file_buffer, cam_final), file_name=f"[{cam_final}] {file_name}", mime="audio/mpeg", key=f"dl_{fid}")
                    
                    df_tl = pd.DataFrame(res['timeline'])
                    df_s = df_tl.sort_values(by="Confiance", ascending=False).reset_index()
                    b_n = df_s.loc[0, 'Note']
                    s_n = df_s[df_s['Note'] != b_n].iloc[0]['Note'] if not df_s[df_s['Note'] != b_n].empty else b_n
                    
                    with c3:
                        st.markdown(f'<div class="metric-container" style="border-bottom: 4px solid #F1C40F;"><div class="label-custom">TOP CONFIANCE</div><div style="font-size:0.85em; margin-top:5px;">🥇 {b_n} <b>({get_camelot_pro(b_n)})</b></div><div style="font-size:0.85em;">🥈 {s_n} <b>({get_camelot_pro(s_n)})</b></div></div>', unsafe_allow_html=True)
                        ct1, ct2 = st.columns(2)
                        with ct1: get_sine_witness(b_n, f"b_{fid}")
                        with ct2: get_sine_witness(s_n, f"s_{fid}")
                    
                    with c4: st.markdown(f'<div class="metric-container"><div class="label-custom">BPM & ENERGIE</div><div class="value-custom">{res["tempo"]}</div><div>E: {res["energy"]}/10</div></div>', unsafe_allow_html=True)

                    st.markdown("---")
                    d1, d2, d3 = st.columns([1, 1, 2])
                    with d1: st.markdown(f"<div class='diag-box'><div class='label-custom'>PURETÉ</div><div style='color:{'#2ECC71' if res['purity'] > 75 else '#F1C40F'}; font-weight:bold;'>{res['purity']}%</div></div>", unsafe_allow_html=True)
                    with d2: st.markdown(f"<div class='diag-box'><div class='label-custom'>MÉTHODE</div><div style='color:#6366F1; font-weight:bold;'>{'✨ HPSS' if res['is_filtered'] else '🎸 DIRECT'}</div></div>", unsafe_allow_html=True)
                    with d3:
                        if res['key_shift']: st.warning(f"Changement détecté : {res['secondary']}")
                        else: st.success("Structure harmonique parfaite.")

                    st.plotly_chart(px.scatter(df_tl, x="Temps", y="Note", color="Confiance", size="Confiance", template="plotly_white"), use_container_width=True)

with tabs[1]:
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)
        st.download_button("📥 TÉLÉCHARGER CSV", df_hist.to_csv(index=False).encode('utf-8'), "historique_ricardo.csv", "text/csv")
    else: st.info("Historique vide.")
