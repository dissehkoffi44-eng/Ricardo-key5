# RCDJ228 SNIPER M3 - VERSION FUSIONNÉE (MOTEUR CODE 2 + ROBUSTESSE CODE 1)
# Avec détection moment modulation + % en target + fin en target
# + Conseils de mix harmonique basés sur la checklist
# + CONSEIL RAPIDE MIX dans le rapport Telegram (version ultra-résumée)

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

# Récupération des secrets
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
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
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4]
    }
}

# --- FONCTIONS UTILITAIRES POUR LES CONSEILS MIX ---

def get_neighbor_camelot(camelot_str: str, offset: int) -> str:
    """Retourne le Camelot voisin avec l'offset donné (modulo 12)"""
    if camelot_str in ['??', None, '']:
        return '??'
    try:
        num = int(camelot_str[:-1])
        wheel = camelot_str[-1]  # A ou B
        new_num = ((num - 1 + offset) % 12) + 1
        return f"{new_num}{wheel}"
    except:
        return '??'

def get_mixing_advice(data):
    """
    Génère les conseils de mix suivant EXACTEMENT la checklist fournie
    """
    if not data.get('modulation', False):
        return None

    principal_camelot = data.get('camelot', '??')
    target_key       = data.get('target_key', 'Inconnu')
    target_camelot   = data.get('target_camelot', '??')
    perc             = data.get('mod_target_percentage', 0)
    ends_in_target   = data.get('mod_ends_in_target', False)
    time_str         = data.get('modulation_time_str', '??:??')

    lines = []

    lines.append("**Checklist mix harmonique – ce que tu dois faire :**")

    if ends_in_target:
        lines.append(f"✅ **Oui : le morceau termine dans {target_key.upper()} ({target_camelot})**")
        lines.append("   → **Privilégie cette tonalité pour le track suivant**")
        priority = "target"
    else:
        lines.append(f"⚠️ **Non : ne termine pas en {target_key.upper()} ({target_camelot})**")
        lines.append("   → La tonalité de sortie reste plutôt " + principal_camelot)
        priority = "principal"

    if perc > 45:
        lines.append(f"✅ **Pourcentage très élevé ({perc:.1f}%)** → traite ce track presque comme s'il était en **{target_camelot}**")
        priority = "target"
    elif perc > 25:
        lines.append(f"ℹ️ **Pourcentage significatif ({perc:.1f}%)** → la target est importante")
        lines.append("   → Tu peux sortir après la bascule pour utiliser la target")
    else:
        lines.append(f"🔸 **Pourcentage faible ({perc:.1f}%)** → modulation plutôt ponctuelle")
        lines.append("   → Tu peux rester sur la tonalité principale pour plus de sécurité")

    lines.append(f"⚠️ **Moment de bascule ≈ {time_str}**")
    lines.append("   → **Évite de faire un long mix pile à cet endroit** (chevauchement de tonalités = risque de clash harmonique)")

    if ends_in_target or perc > 40:
        lines.append("")
        lines.append("**🚀 Pour une montée d’énergie volontaire :**")
        lines.append(f"   → Sors sur la fin → enchaîne sur un track **+3** ou **+7** depuis **{target_camelot}**")
        lines.append(f"     Ex : {target_camelot} → **{get_neighbor_camelot(target_camelot, 3)}** ou **{get_neighbor_camelot(target_camelot, 7)}**")
        lines.append("     → C’est une vraie « modulation DJ » qui donne du punch !")

    lines.append("")
    lines.append("**Choix le plus safe pour le track suivant :**")
    if priority == "target":
        lines.append(f"→ **{target_camelot}** ou voisins (±1 sur la même roue A/B)")
    else:
        lines.append(f"→ **{principal_camelot}** ou voisins (±1)")

    return "\n".join(lines)

# --- AJOUTS POUR PIANO COMPANION INTÉGRATION ---

def get_diatonic_chords(key):
    """Génère les accords diatoniques pour une tonalité, comme dans Piano Companion."""
    if not key or key == "Unknown":
        return []
    
    note, mode = key.split()
    root_idx = NOTES_LIST.index(note)
    
    if mode == 'major':
        scale_intervals = [0, 2, 4, 5, 7, 9, 11]  # Gamme majeure
        chord_types = ['maj', 'min', 'min', 'maj', 'maj', 'min', 'dim']
        roman_numerals = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°']
    else:  # minor (naturelle pour simplicité, comme base dans l'app)
        scale_intervals = [0, 2, 3, 5, 7, 8, 10]  # Gamme mineure naturelle
        chord_types = ['min', 'dim', 'maj', 'min', 'min', 'maj', 'maj']
        roman_numerals = ['i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII']
    
    scale_notes = [NOTES_LIST[(root_idx + interv) % 12] for interv in scale_intervals]
    
    chords = []
    for i in range(7):
        chord_root = scale_notes[i]
        ctype = chord_types[i]
        if ctype == 'maj':
            third_interval = 4
            fifth_interval = 7
        elif ctype == 'min':
            third_interval = 3
            fifth_interval = 7
        elif ctype == 'dim':
            third_interval = 3
            fifth_interval = 6
        third = NOTES_LIST[(NOTES_LIST.index(chord_root) + third_interval) % 12]
        fifth = NOTES_LIST[(NOTES_LIST.index(chord_root) + fifth_interval) % 12]
        chord_notes = [chord_root, third, fifth]
        chords.append({
            'roman': roman_numerals[i],
            'name': f"{chord_root}{ctype}",
            'notes': ' '.join(chord_notes),
            'notes_list': chord_notes
        })
    
    return chords

def get_diatonic_notes(key):
    """Retourne les notes uniques diatoniques d'une tonalité (pour comparaison)."""
    chords = get_diatonic_chords(key)
    if not chords:
        return set()
    diat_notes = set()
    for chord in chords:
        diat_notes.update(chord['notes'].split())
    return diat_notes

def test_chord_consonance(chroma_norm, chord_notes_list):
    """Teste la consonance d'un accord en sommant les valeurs chroma normalisées de ses notes."""
    indices = [NOTES_LIST.index(note) for note in chord_notes_list]
    consonance_score = np.sum([chroma_norm[idx] for idx in indices]) / len(indices)  # Moyenne pour normalisation
    return consonance_score

# --- STYLES CSS ---
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
        background: rgba(239, 68, 68, 0.20); color: #fca5a5;
        padding: 18px; border-radius: 12px; border: 1px solid #ef4444;
        margin: 20px 0; font-weight: bold; font-family: 'JetBrains Mono', monospace;
        line-height: 1.6; font-size: 1.05em;
    }
    .modulation-alert .detail {
        color: #fbbf24; font-size: 1.1em;
    }
    .modulation-alert .nature {
        font-size: 0.92em; opacity: 0.85; color: #fca5a5;
    }
    .metric-box {
        background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d;
        height: 100%; transition: 0.3s;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE CALCUL ---
def apply_sniper_filters(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    nyq = 0.5 * sr
    low = 80 / nyq
    high = 5000 / nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y_harm)

def get_bass_priority(y, sr):
    nyq = 0.5 * sr
    b, a = butter(2, 150/nyq, btype='low')
    y_bass = lfilter(b, a, y)
    chroma_bass = librosa.feature.chroma_cqt(y=y_bass, sr=sr, n_chroma=12)
    return np.mean(chroma_bass, axis=1)

def solve_key_sniper(chroma_vector, bass_vector):
    best_overall_score = -1
    best_key = "Unknown"
    
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)
    
    key_scores = {f"{NOTES_LIST[i]} {mode}": [] for mode in ["major", "minor"] for i in range(12)}
    
    for p_name, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                
                dom_idx = (i + 7) % 12
                leading_tone = (i + 11) % 12
                
                if mode == "minor":
                    if cv[leading_tone] > 0.30:
                        score *= 1.35
                    if cv[dom_idx] > 0.45:
                        score *= 1.15
                else:
                    if cv[i] > 0.7 and cv[dom_idx] > 0.6:
                        score *= 1.1
                
                if bv[i] > 0.6:
                    score += (bv[i] * 0.25)
                
                third_idx = (i + 4) % 12 if mode == "major" else (i + 3) % 12
                if cv[third_idx] > 0.5:
                    score += 0.15
                
                fifth_idx = (i + 7) % 12
                if cv[fifth_idx] > 0.5:
                    score += 0.10
                
                key_name = f"{NOTES_LIST[i]} {mode}"
                key_scores[key_name].append(score)
    
    for key_name, scores in key_scores.items():
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_overall_score:
                best_overall_score = avg_score
                best_key = key_name
    
    return {"key": best_key, "score": best_overall_score}

def seconds_to_mmss(seconds):
    if seconds is None:
        return "??:??"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def process_audio_precision(file_bytes, file_name, _progress_callback=None):
    ext = file_name.split('.')[-1].lower()
    try:
        if ext == 'm4a':
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            y = samples / (2**15)
            sr = audio.frame_rate
            if sr != 22050:
                y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                sr = 22050
        else:
            with io.BytesIO(file_bytes) as buf:
                y, sr = librosa.load(buf, sr=22050, mono=True)
    except Exception as e:
        st.error(f"Erreur de lecture du fichier {file_name}: {e}")
        return None

    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_sniper_filters(y, sr)

    step, timeline, votes = 6, [], Counter()
    segments = list(range(0, max(1, int(duration) - step), 2))
    total_segments = len(segments)
    
    for idx, start in enumerate(segments):
        if _progress_callback:
            prog_internal = int((idx / total_segments) * 100)
            _progress_callback(prog_internal, f"Scan : {start}s / {int(duration)}s")

        idx_start, idx_end = int(start * sr), int((start + step) * sr)
        seg = y_filt[idx_start:idx_end]
        if len(seg) < 1000 or np.max(np.abs(seg)) < 0.01: continue
        
        c_raw = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=24, bins_per_octave=24)
        c_avg = np.mean((c_raw[::2, :] + c_raw[1::2, :]) / 2, axis=1)
        b_seg = get_bass_priority(y[idx_start:idx_end], sr)
        res = solve_key_sniper(c_avg, b_seg)
        
        if res['score'] < 0.9: continue
        
        weight = 2.0 if (start < 10 or start > (duration - 15)) else 1.0
        votes[res['key']] += int(res['score'] * 100 * weight)
        timeline.append({"Temps": start, "Note": res['key'], "Conf": res['score']})

    if not votes:
        return None

    most_common = votes.most_common(2)
    final_key = most_common[0][0]
    final_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == final_key]) * 100)
    
    mod_detected = len(most_common) > 1 and (votes[most_common[1][0]] / max(1, sum(votes.values()))) > 0.25
    target_key = most_common[1][0] if mod_detected else None

    modulation_time = None
    target_percentage = 0
    ends_in_target = False

    if mod_detected and target_key:
        candidates = [t["Temps"] for t in timeline if t["Note"] == target_key and t["Conf"] >= 0.84]
        if candidates:
            modulation_time = min(candidates)
        else:
            target_times = [t["Temps"] for t in timeline if t["Note"] == target_key]
            if target_times:
                sorted_times = sorted(target_times)
                modulation_time = sorted_times[max(0, len(sorted_times) // 3)]

        total_valid = len(timeline)
        if total_valid > 0:
            target_count = sum(1 for t in timeline if t["Note"] == target_key)
            final_count = sum(1 for t in timeline if t["Note"] == final_key)
            target_percentage = (target_count / total_valid) * 100

        if timeline:
            last_n = max(5, len(timeline) // 10)
            last_segments = timeline[-last_n:]
            last_counter = Counter(s["Note"] for s in last_segments)
            last_key = last_counter.most_common(1)[0][0]
            ends_in_target = (last_key == target_key)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma_avg = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)

    # --- AJOUT : Comparaison avec top 5 notes dominantes pour décision finale ---
    chroma_norm = chroma_avg / np.max(chroma_avg)
    top_indices = np.argsort(chroma_norm)[-5:]  # Top 5 notes
    top_notes = set(NOTES_LIST[i] for i in top_indices)

    # Candidats : Top 2 des votes (fidèle à l'original)
    candidates = [mc[0] for mc in most_common]

    # Score match : Proportion de top notes dans diatoniques (pondéré 0.3 sur conf)
    matches = {}
    for key in candidates:
        diat_notes = get_diatonic_notes(key)
        match_count = len(top_notes.intersection(diat_notes)) / 5
        # Bonus si tierce match mode
        note, mode = key.split()
        root_idx = NOTES_LIST.index(note)
        third_idx = (root_idx + 4 if mode == 'major' else root_idx + 3) % 12
        if NOTES_LIST[third_idx] in top_notes:
            match_count += 0.2  # Bonus tierce
        matches[key] = match_count

    # Ajuste conf avec match (pondéré)
    best_key = max(matches, key=matches.get)
    best_match = matches[best_key]
    final_conf = int(final_conf * (0.7 + 0.3 * best_match))  # Ajuste conf
    final_key = best_key  # Switch si meilleur match

    # --- AJOUT : Test de consonance sur les accords diatoniques ---
    diatonic_chords = get_diatonic_chords(final_key)
    consonance_scores = {}
    for chord in diatonic_chords:
        score = test_chord_consonance(chroma_norm, chord['notes_list'])
        consonance_scores[chord['name']] = round(score * 100, 1)  # Pour % avec 1 décimale

    # Meilleur accord (celui avec le score le plus haut)
    if consonance_scores:
        best_chord_name = max(consonance_scores, key=consonance_scores.get)
        best_chord_score = consonance_scores[best_chord_name]
    else:
        best_chord_name = "None"
        best_chord_score = 0

    # --- AJOUT : Meilleur accord global (tous les triads possibles, excluant root de main/target si applicable) ---
    all_chords = []
    for root in NOTES_LIST:
        # Major
        third_maj = NOTES_LIST[(NOTES_LIST.index(root) + 4) % 12]
        fifth_maj = NOTES_LIST[(NOTES_LIST.index(root) + 7) % 12]
        all_chords.append({
            'name': f"{root}maj",
            'notes_list': [root, third_maj, fifth_maj]
        })
        # Minor
        third_min = NOTES_LIST[(NOTES_LIST.index(root) + 3) % 12]
        fifth_min = NOTES_LIST[(NOTES_LIST.index(root) + 7) % 12]
        all_chords.append({
            'name': f"{root}min",
            'notes_list': [root, third_min, fifth_min]
        })
        # Dim
        third_dim = NOTES_LIST[(NOTES_LIST.index(root) + 3) % 12]
        fifth_dim = NOTES_LIST[(NOTES_LIST.index(root) + 6) % 12]
        all_chords.append({
            'name': f"{root}dim",
            'notes_list': [root, third_dim, fifth_dim]
        })

    overall_consonance_scores = {}
    for chord in all_chords:
        score = test_chord_consonance(chroma_norm, chord['notes_list'])
        overall_consonance_scores[chord['name']] = round(score * 100, 1)

    # Identifier les chords à exclure (root de main et target)
    exclude_chords = set()
    note, mode = final_key.split()
    root_chord = f"{note}maj" if mode == 'major' else f"{note}min"
    exclude_chords.add(root_chord)
    if target_key:
        t_note, t_mode = target_key.split()
        target_root_chord = f"{t_note}maj" if t_mode == 'major' else f"{t_note}min"
        exclude_chords.add(target_root_chord)

    # Trier et trouver le meilleur non exclu
    sorted_chords = sorted(overall_consonance_scores.items(), key=lambda x: x[1], reverse=True)
    best_global_chord = None
    best_global_score = 0
    for ch, sc in sorted_chords:
        if ch not in exclude_chords:
            best_global_chord = ch
            best_global_score = sc
            break

    if not best_global_chord:
        best_global_chord = "None"
        best_global_score = 0

    # Génération accords pour affichage (sur final_key ajustée)
    diatonic_chords = get_diatonic_chords(final_key)
    target_diatonic_chords = get_diatonic_chords(target_key) if target_key else []

    # Validation score pour affichage (coverage globale)
    validation_score = best_match * 100

    res_obj = {
        "key": final_key,
        "camelot": CAMELOT_MAP.get(final_key, "??"),
        "conf": min(final_conf, 99),
        "tempo": int(float(tempo)),
        "tuning": round(440 * (2**(tuning/12)), 1),
        "timeline": timeline,
        "chroma": chroma_avg.tolist(),
        "modulation": mod_detected,
        "target_key": target_key,
        "target_camelot": CAMELOT_MAP.get(target_key, "??") if target_key else None,
        "modulation_time_str": seconds_to_mmss(modulation_time) if mod_detected else None,
        "mod_target_percentage": round(target_percentage, 1) if mod_detected else 0,
        "mod_ends_in_target": ends_in_target if mod_detected else False,
        "name": file_name,
        "diatonic_chords": diatonic_chords,
        "target_diatonic_chords": target_diatonic_chords,
        "validation_score": int(validation_score),
        "key_alternatives": [k for k in candidates if k != final_key],
        "best_chord": best_chord_name,  # Best diatonic
        "best_chord_consonance": int(best_chord_score),
        "best_global_chord": best_global_chord,  # Best global excluant roots
        "best_global_consonance": int(best_global_score),
        "consonance_scores": consonance_scores,
        "overall_consonance_scores": overall_consonance_scores
    }

    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            df_tl = pd.DataFrame(timeline)
            fig_tl = px.line(df_tl, x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER})
            img_tl = fig_tl.to_image(format="png", width=1000, height=500)
            
            fig_rd = go.Figure(data=go.Scatterpolar(r=res_obj['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
            fig_rd.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
            img_rd = fig_rd.to_image(format="png", width=600, height=600)

            mod_line = ""
            if mod_detected:
                perc = res_obj["mod_target_percentage"]
                end_txt = " → **fin en " + target_key.upper() + "**" if res_obj['mod_ends_in_target'] else ""
                mod_line = f"  *MODULATION →* `{target_key.upper()}` ({res_obj['target_camelot']}) ≈ **{res_obj['modulation_time_str']}** ({perc}%){end_txt}"

            caption = (f"  *RCDJ228 MUSIC SNIPER - RAPPORT*\n━━━━━━━━━━━━\n"
                       f"  *FICHIER:* `{file_name}`\n"
                       f"  *TONALITÉ:* `{final_key.upper()}`\n"
                       f"  *CAMELOT:* `{res_obj['camelot']}`\n"
                       f"  *CONFIANCE:* `{res_obj['conf']}%`\n"
                       f"  *TEMPO:* `{res_obj['tempo']} BPM`\n"
                       f"  *ACCORDAGE:* `{res_obj['tuning']} Hz`\n"
                       f"{mod_line if mod_detected else '  *STABILITÉ TONALE:* OK'}\n━━━━━━━━━━━━")

            # ─── AJOUT : CONSEIL RAPIDE MIX EN VERSION ULTRA-RÉSUMÉE ───
            advice_text = get_mixing_advice(res_obj)
            summary_advice = ""
            if advice_text:
                if "fin en target" in advice_text or "Oui : le morceau termine" in advice_text:
                    summary_advice = f"→ Termine en {res_obj['target_camelot']} → privilégie cette tonalité pour le suivant !"
                elif res_obj.get('mod_target_percentage', 0) > 45:
                    summary_advice = f"→ {res_obj['target_camelot']} très présent → traite presque comme track en {res_obj['target_camelot']}"
                else:
                    summary_advice = "→ Modulation ponctuelle → reste sur tonalité principale"
            
            if summary_advice:
                caption += f"\n\n*Conseil rapide mix :* {summary_advice}"
            else:
                caption += "\n\n*Pas de modulation détectée → mix safe sur la tonalité principale*"

            files = {'p1': ('timeline.png', img_tl, 'image/png'), 'p2': ('radar.png', img_rd, 'image/png')}
            media = [
                {'type': 'photo', 'media': 'attach://p1', 'caption': caption, 'parse_mode': 'Markdown'},
                {'type': 'photo', 'media': 'attach://p2'}
            ]
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMediaGroup",
                data={'chat_id': CHAT_ID, 'media': json.dumps(media)},
                files=files,
                timeout=20
            )
        except Exception:
            pass

    del y, y_filt
    gc.collect()
    return res_obj

def get_chord_js(btn_id, key_str):
    note, mode = key_str.split()
    return f"""
    document.getElementById('{btn_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
        const intervals = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach(i => {{
            const o = ctx.createOscillator(); const g = ctx.createGain();
            o.type = 'triangle'; 
            o.frequency.setValueAtTime(freqs['{note}'] * Math.pow(2, i/12), ctx.currentTime);
            g.gain.setValueAtTime(0, ctx.currentTime);
            g.gain.linearRampToValueAtTime(0.15, ctx.currentTime + 0.1);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 2.0);
            o.connect(g); g.connect(ctx.destination);
            o.start(); o.stop(ctx.currentTime + 2.0);
        }});
    }}; """

# ────────────────────────────────────────────────
#               INTERFACE PRINCIPALE
# ────────────────────────────────────────────────

st.title("🎯 RCDJ228 MUSIC SNIPER")

uploaded_files = st.file_uploader(
    "Déposez vos fichiers audio (mp3, wav, flac, m4a)",
    type=['mp3','wav','flac','m4a'],
    accept_multiple_files=True
)

if uploaded_files:
    global_progress_placeholder = st.empty()
    total_files = len(uploaded_files)
    results_container = st.container()
    
    for i, f in enumerate(reversed(uploaded_files)):
        global_progress_placeholder.markdown(f"""
            <div style="padding:15px; border-radius:15px; background-color:rgba(16,185,129,0.12); border:1px solid #10b981; margin-bottom:20px;">
                <h3 style="margin:0; color:#10b981;">Analyse en cours : {i+1} / {total_files}</h3>
                <p style="margin:6px 0 0 0; opacity:0.85;">{f.name}</p>
            </div>
            """, unsafe_allow_html=True)

        with st.status(f"Analyse → {f.name}", expanded=True) as status:
            inner_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(val, msg):
                inner_bar.progress(val)
                status_text.code(msg)

            audio_bytes = f.getvalue()
            data = process_audio_precision(audio_bytes, f.name, update_progress)
            status.update(label=f"{f.name} terminé", state="complete", expanded=False)

        if data:
            with results_container:
                st.markdown(f"<div class='file-header'> {data['name']}</div>", unsafe_allow_html=True)
                
                key_col, chord_col = st.columns(2)

                with key_col:
                    color = "linear-gradient(135deg, #065f46, #064e3b)" if data['conf'] > 85 else "linear-gradient(135deg, #1e293b, #0f172a)"

                    mod_alert = ""
                    if data.get('modulation'):
                        perc = data.get('mod_target_percentage', 0)
                        ends_in_target = data.get('mod_ends_in_target', False)
                        time_str = data.get('modulation_time_str', '??:??')
                        
                        if perc < 25:
                            nature = "passage court / ponctuel"
                        elif perc < 50:
                            nature = "section significative"
                        else:
                            nature = "dominante sur une grande partie"

                        end_txt = " → **fin en " + data['target_key'].upper() + "**" if ends_in_target else ""
                        
                        mod_alert = f"""
                            <div class="modulation-alert">
                                MODULATION → {data['target_key'].upper()} ({data['target_camelot']})<br>
                                <span class="detail">≈ {time_str} – {perc}% du morceau{end_txt}</span><br>
                                <span class="nature">({nature})</span>
                            </div>
                        """.strip()

                    st.markdown(f"""
                        <div class="report-card" style="background:{color};">
                            <h1 style="font-size:5.4em; margin:8px 0; font-weight:900;">{data['key'].upper()}</h1>
                            <p style="font-size:1.5em; opacity:0.92;">
                                CAMELOT <b>{data['camelot']}</b>  •  Confiance <b>{data['conf']}%</b>
                            </p>
                            {mod_alert}
                        </div>
                        """, unsafe_allow_html=True)

                with chord_col:
                    color_chord = "linear-gradient(135deg, #4338ca, #3730a3)" if data['best_global_consonance'] > 85 else "linear-gradient(135deg, #1e293b, #0f172a)"
                    st.markdown(f"""
                        <div class="report-card" style="background:{color_chord};">
                            <h1 style="font-size:5.4em; margin:8px 0; font-weight:900;">{data['best_global_chord'].upper()}</h1>
                            <p style="font-size:1.5em; opacity:0.92;">
                                MEILLEUR ACCORD  •  Consonance <b>{data['best_global_consonance']}%</b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                advice = get_mixing_advice(data)
                if advice:
                    with st.expander("📋 Checklist MIX – que faire avec ce track ?", expanded=True):
                        st.markdown(
                            f"""
                            <div style="
                                background: linear-gradient(135deg, rgba(16,185,129,0.10), rgba(16,185,129,0.04));
                                border: 1px solid rgba(16,185,129,0.4);
                                border-radius: 12px;
                                padding: 20px 24px;
                                margin: 16px 0;
                                line-height: 1.65;
                                font-size: 1.02em;
                                white-space: pre-wrap;
                            ">
                            {advice}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                m1, m2, m3 = st.columns(3)
                with m1: 
                    st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2.4em; color:#10b981;'>{data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
                with m2: 
                    st.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br><span style='font-size:2.2em; color:#58a6ff;'>{data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)
                with m3:
                    btn_id = f"play_{i}_{hash(data['name'])}"
                    components.html(f"""
                        <button id="{btn_id}" style="width:100%; height:95px; background:linear-gradient(45deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold; font-size:1.1em;">
                            TESTER L'ACCORD
                        </button>
                        <script>{get_chord_js(btn_id, data['key'])}</script>
                    """, height=110)

                c1, c2 = st.columns([2, 1])
                with c1: 
                    fig_tl = px.line(
                        pd.DataFrame(data['timeline']), 
                        x="Temps", y="Note", 
                        markers=True, 
                        template="plotly_dark", 
                        category_orders={"Note": NOTES_ORDER}
                    )
                    fig_tl.update_layout(height=320, margin=dict(l=0,r=0,t=20,b=0))
                    st.plotly_chart(fig_tl, use_container_width=True, key=f"tl_{i}_{hash(f.name)}")
                
                with c2: 
                    fig_rd = go.Figure(data=go.Scatterpolar(
                        r=data['chroma'], 
                        theta=NOTES_LIST, 
                        fill='toself', 
                        line_color='#10b981'
                    ))
                    fig_rd.update_layout(
                        template="plotly_dark", 
                        height=320, 
                        polar=dict(radialaxis=dict(visible=False)),
                        margin=dict(l=30,r=30,t=20,b=20)
                    )
                    st.plotly_chart(fig_rd, use_container_width=True, key=f"rd_{i}_{hash(f.name)}")
                
                # --- AJOUT : Affichage des accords et validation ---
                with st.expander("🎹 Accords diatoniques (comme dans Piano Companion) – pour valider la tonalité", expanded=False):
                    chords = data.get("diatonic_chords", [])
                    if chords:
                        df_chords = pd.DataFrame(chords)
                        st.table(df_chords)  # Premier tableau : accords diatoniques
                        st.markdown(f"**Score de validation (coverage chroma) :** {data.get('validation_score', 0)}%")
                        if data.get('key_alternatives'):
                            st.warning(f"Alternatives possibles : {', '.join(data['key_alternatives'])} – Vérifie manuellement !")
                    else:
                        st.info("Pas d'accords générés (tonalité inconnue).")
                    
                    # Pour la modulation
                    target_chords = data.get("target_diatonic_chords", [])
                    if target_chords:
                        st.subheader("Accords pour la tonalité cible (modulation)")
                        df_target = pd.DataFrame(target_chords)
                        st.table(df_target)
                    
                    # Deuxième tableau : scores de consonance pour les accords diatoniques
                    if consonance_scores:
                        st.subheader("Scores de consonance des accords diatoniques")
                        consonance_data = [{'Accord': name, 'Score (%)': score} for name, score in sorted(consonance_scores.items(), key=lambda x: x[1], reverse=True)]
                        df_consonance = pd.DataFrame(consonance_data)
                        st.table(df_consonance)
                    
                    # Affichage du meilleur accord
                    st.markdown(f"**Meilleur accord consonant :** {data.get('best_chord', 'None')} (Score: {data.get('best_chord_consonance', 0)}%)")
                
                st.markdown("<hr style='border-color:#30363d; margin:40px 0 30px 0;'>", unsafe_allow_html=True)

    global_progress_placeholder.success(f"Analyse terminée — {total_files} piste(s) traitée(s) avec succès !")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2569/2569107.png", width=80)
    st.header("Contrôles Sniper")
    if st.button("🔄 Vider le cache & relancer"):
        st.cache_data.clear()
        st.rerun()
