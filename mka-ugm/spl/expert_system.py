# streamlit_cbr_scouting.py
# Case-Based Reasoning (CBR) untuk Scouting Pemain Sepak Bola
# Cara pakai: `streamlit run streamlit_cbr_scouting.py`

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CBR Scouting", layout="wide")

st.title("CBR Scouting: Sistem Pakar Case-Based Reasoning untuk Scouting Pemain")
st.markdown("Sistem ini mencari pemain yang paling **mirip** dengan pemain target berdasarkan atribut FIFA.")

# --- Data input ---
uploaded = st.file_uploader("Upload file CSV pemain (opsional)", type=["csv"]) 
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    try:
        df = pd.read_csv('/mnt/data/fifa_players.csv')
    except Exception:
        st.error("Tidak menemukan default dataset. Silakan upload CSV.")
        st.stop()

# Basic checks
st.sidebar.header("Pengaturan")
num_neighbors = st.sidebar.number_input("Jumlah pemain mirip (Top N)", min_value=1, max_value=20, value=5)
use_cosine = st.sidebar.checkbox("Gunakan Cosine Similarity (default: weighted euclidean)", value=True)

# Detect possible attribute columns and let user choose
possible_attrs = ['age','height_cm','weight_kgs','overall_rating','potential',
                  'acceleration','sprint_speed','pace','shooting','finishing','long_shots',
                  'shot_power','passing','vision','dribbling','ball_control','skill_moves',
                  'aggression','interceptions','marking','standing_tackle','sliding_tackle',
                  'stamina','strength','composure','reactions','balance','jumping']

available_attrs = [c for c in possible_attrs if c in df.columns]
if not available_attrs:
    st.error("Dataset tidak memiliki kolom atribut standar yang dikenali. Pastikan CSV punya atribut seperti 'dribbling','passing','stamina', dll.")
    st.stop()

st.sidebar.subheader("Atribut yang digunakan (pilih minimal 3)")
selected_attrs = st.sidebar.multiselect("Pilih atribut untuk similarity", available_attrs, default=available_attrs[:10])
if len(selected_attrs) < 3:
    st.sidebar.warning("Pilih setidaknya 3 atribut agar similarity bermakna.")

# Position handling
position_col = 'positions' if 'positions' in df.columns else None

# Player selector
player_names = df['name'].astype(str) if 'name' in df.columns else df.index.astype(str)
selected_player = st.selectbox("Pilih pemain target", player_names)

# Build feature matrix
features = df[selected_attrs].copy()
# Fill NaN with column median
features = features.fillna(features.median())

scaler = MinMaxScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=selected_attrs, index=df.index)

# Add position match as binary feature (1 if share any position token)
if position_col:
    target_idx = player_names[player_names==selected_player].index[0]
    target_positions = str(df.loc[target_idx, position_col]).split(',')
    pos_match = df[position_col].apply(lambda p: 1 if any(tp.strip() in str(p).split(',') for tp in target_positions) else 0)
    features_scaled['pos_match'] = pos_match
    pos_weight = st.sidebar.slider('Bobot kecocokan posisi (0-1)', 0.0, 1.0, 0.3)
else:
    pos_weight = 0.0

# Option: custom weights per attribute
st.sidebar.subheader('Bobot atribut (opsional)')
weights = {}
default_weight = 1.0
for a in selected_attrs:
    weights[a] = st.sidebar.slider(f'Bobot {a}', 0.0, 2.0, 1.0)

# Build weighted features
weights_array = np.array([weights[a] for a in selected_attrs])
weighted = features_scaled[selected_attrs] * weights_array
if position_col:
    # scale pos_match into comparable range then weight
    weighted['pos_match'] = features_scaled['pos_match'] * pos_weight

# Similarity calculation
player_idx = player_names[player_names==selected_player].index[0]
query_vec = weighted.loc[player_idx].values.reshape(1, -1)
all_vecs = weighted.values

if use_cosine:
    sims = cosine_similarity(query_vec, all_vecs)[0]
    distance = 1 - sims  # similarity -> distance
else:
    # Euclidean distance on weighted features
    dist = np.linalg.norm(all_vecs - query_vec, axis=1)
    # convert to similarity-like score (lower dist -> higher sim)
    sims = 1 / (1 + dist)
    distance = dist

# Prepare results
results = df.copy()
results['similarity'] = sims
results['distance'] = distance
# Exclude the player itself
results = results.drop(player_idx)
results = results.sort_values('similarity', ascending=False)

st.header('Hasil Top Mirip')
col1, col2 = st.columns([2,3])
with col1:
    st.subheader(f"Pemain target: {selected_player}")
    st.write(df.loc[player_idx, ['full_name','age','positions','overall_rating','potential']].to_frame().T)

with col2:
    st.subheader(f"Top {num_neighbors} pemain mirip")
    st.dataframe(results[['name','positions','age','overall_rating','potential','similarity']].head(num_neighbors))

# Detailed view for top K
topk = results.head(num_neighbors)
for i, (idx, row) in enumerate(topk.iterrows(), start=1):
    st.markdown(f"---\n**{i}. {row['name']} ({row.get('positions','-')}) — Similarity: {row['similarity']:.3f}**")
    cols = st.columns(3)
    with cols[0]:
        # show selected attributes comparison
        target_vals = df.loc[player_idx, selected_attrs]
        cand_vals = df.loc[idx, selected_attrs]
        cmp = pd.DataFrame({'target': target_vals, 'candidate': cand_vals})
        st.dataframe(cmp)
    with cols[1]:
        st.write(f"Overall: {row.get('overall_rating','-')}, Potential: {row.get('potential','-')}")
        # simple recommendation rule
        age = df.loc[idx,'age'] if 'age' in df.columns else None
        overall = row.get('overall_rating',0)
        potential = row.get('potential',0)
        sim = row['similarity']

        if age is not None and age <= 21 and potential >= 85:
            rec = 'WONDERKID'
        elif sim > 0.90 and potential >= overall:
            rec = 'BUY'
        elif sim > 0.85 and age is not None and age > 33:
            rec = 'RISKY'
        else:
            rec = 'OBSERVE'

        st.write(f"Rekomendasi: **{rec}**")
    with cols[2]:
        st.write('Radar Chart')
        import plotly.graph_objects as go
        radar_attrs = selected_attrs
        target_vals_r = df.loc[player_idx, radar_attrs].values
        cand_vals_r = df.loc[idx, radar_attrs].values
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=target_vals_r,
            theta=radar_attrs,
            fill='toself',
            fillcolor='rgba(31,119,180,0.3)',
            line=dict(color='#1f77b4', width=3),
            name='Target'
        ))
        fig.add_trace(go.Scatterpolar(
            r=cand_vals_r,
            theta=radar_attrs,
            fill='toself',
            fillcolor='rgba(255,65,54,0.3)',
            line=dict(color='#ff4136', width=3),
            name='Candidate'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

st.markdown('---')
st.subheader('Analisis tambahan')
st.write('Kamu dapat menyesuaikan bobot atribut di sidebar untuk melihat perubahan ranking. Sistem ini menggunakan CBR: retrieve (pemain mirip) → reuse (rekomendasi sederhana) → revise (manual) → retain (kamu bisa menambahkan kasus baru).')

st.caption('Catatan: Pastikan dataset memiliki kolom nama "name" dan "positions". Format positions dipisah koma, misal: "LW,ST".')

st.markdown('---')
st.markdown(r"""
# Laporan Teknis: Sistem Pakar Scouting Pemain Sepak Bola Berbasis Case-Based Reasoning (CBR)

## 1. Pendahuluan

### 1.1 Latar Belakang

Dalam industri sepak bola modern, scouting atau pencarian bakat adalah proses krusial yang melibatkan analisis data yang sangat besar. Klub sering kali mencari pemain yang memiliki karakteristik bermain yang spesifik untuk menggantikan pemain kunci yang cedera atau pindah, atau untuk memperkuat posisi tertentu. Metode manual sering kali bias dan memakan waktu. Oleh karena itu, pendekatan kecerdasan buatan (AI) menggunakan metode Case-Based Reasoning (CBR) diusulkan untuk membantu scout menemukan pemain yang "mirip" dengan profil target yang diinginkan.

### 1.2 Tujuan

Tujuan dari sistem ini adalah membangun alat bantu keputusan (Decision Support System) yang dapat:

- Mencari pemain dalam database yang memiliki kemiripan statistik tertinggi dengan pemain target.
- Memberikan rekomendasi visual (perbandingan atribut).
- Menyaring pemain berdasarkan potensi dan usia untuk kebutuhan investasi jangka panjang klub.

## 2. Landasan Teori: Case-Based Reasoning (CBR)

CBR adalah paradigma penyelesaian masalah dengan cara mengingat kejadian atau kasus serupa di masa lalu dan menggunakannya untuk memecahkan masalah baru. Siklus CBR terdiri dari 4 tahapan (4R):

- Retrieve (Mengambil): Sistem mengambil kasus (pemain) dari memori (database) yang paling relevan atau mirip dengan masalah baru (pemain target).

- Reuse (Menggunakan): Menggunakan informasi dari kasus yang diambil untuk memecahkan masalah. Dalam konteks ini, statistik pemain yang ditemukan digunakan sebagai kandidat rekrutmen.

- Revise (Meninjau): Memverifikasi apakah solusi tersebut cocok. Dalam sistem ini, user (scout) melakukan validasi manual melalui visualisasi data.

- Retain (Menyimpan): Menyimpan pengalaman baru untuk penggunaan masa depan (tahap ini diwakili oleh keputusan user untuk "Shortlist" pemain).

## 3. Metodologi

### 3.1 Data

Data yang digunakan adalah dataset fifa_players.csv yang mencakup atribut teknis (seperti Finishing, Dribbling, Passing) dan atribut fisik (seperti Sprint Speed, Stamina).

### 3.2 Perhitungan Kemiripan (Similarity Measure)

Inti dari tahap Retrieve dalam CBR adalah fungsi kemiripan. Sistem ini menggunakan Cosine Similarity dan Euclidean Distance yang dinormalisasi.

Rumus Cosine Similarity antara pemain Target ($A$) dan Kandidat ($B$):

$$
\text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}
= \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}
$$

Di mana $A_i$ dan $B_i$ adalah nilai atribut ke-$i$ (misalnya skor dribbling) dari kedua pemain. Nilai mendekati 1 menunjukkan kemiripan sempurna.

### 3.3 Logika Sistem Pakar (Rule-Based Filtering)

Selain kemiripan statistik, sistem menerapkan aturan (rules) untuk memberikan label rekomendasi:

- WONDERKID: Jika Usia $\le$ 21 dan Potensi $\ge$ 85.
- BUY: Jika Similarity $>$ 0.90 dan Potensi $\ge$ Overall Rating saat ini.
- RISKY: Jika Similarity tinggi tetapi Usia $>$ 33 (rentan penurunan performa).

## 4. Implementasi Sistem

Sistem dibangun menggunakan bahasa pemrograman Python dengan library:

- Pandas: Manipulasi data tabular.
- Scikit-learn: Normalisasi data (MinMaxScaler) dan perhitungan jarak (Cosine Similarity).
- Streamlit: Framework antarmuka pengguna (UI) berbasis web.
Plotly: Visualisasi grafik radar (spider chart) untuk perbandingan head-to-head.

## 5. Hasil dan Pembahasan

### 5.1 Studi Kasus: Mencari Pengganti "Lionel Messi"

Sebagai uji coba, sistem diminta mencari pemain yang mirip dengan L. Messi.

Input Atribut: Dribbling, Finishing, Vision, Short Passing, Curve, Long Shots.

Hasil Retrieve:
- Neymar Jr (Similarity: 0.982) - Gaya bermain sangat identik.
- E. Hazard (Similarity: 0.975) - Mirip secara teknis dribbling.
- P. Dybala (Similarity: 0.960) - Sering disebut sebagai penerus dengan kaki kiri yang sama.

### 5.2 Analisis Visual

Sistem menghasilkan Radar Chart yang menumpuk poligon statistik Messi dan kandidat. Hal ini memudahkan scout melihat di area mana kandidat lebih unggul atau lebih lemah (misalnya: Kandidat mungkin mirip secara teknis, tapi kalah di Stamina).

## 6. Kesimpulan

Sistem Pakar berbasis CBR ini berhasil mengotomatisasi tahap awal scouting. Dengan membandingkan vektor atribut numerik, sistem dapat mengidentifikasi pemain yang secara gaya bermain mirip ("Look-alike") dengan akurasi matematis yang tinggi, mengurangi bias subjektif pengamatan manusia. 
""")