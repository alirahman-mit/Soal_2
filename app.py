# =============================================================================
# app.py — Aplikasi Deteksi Gambar CNN (Single-Page)
# Proyek  : Week 2 — Image Classification with CNN
# Model   : Custom CNN (Apel vs Jeruk) | VGG16 Fine-tuned (Penyakit Daun Tomat)
# =============================================================================

import os
import numpy as np
import streamlit as st
from PIL import Image
import gdown

from utils import load_custom_css, load_tf_model, preprocess_image

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN (WAJIB PALING ATAS & CUMA BOLEH 1 KALI!)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1. Pastikan folder models tersedia
if not os.path.exists('models'):
    os.makedirs('models')

# 2. Fungsi download model dari Google Drive (hanya jika file belum ada)
def download_models():

    path_model_1 = 'models/Model_Custom_ApelJeruk.h5'
    path_model_2 = 'models/Model_VGG16_Tomat.h5'

    # ID Google Drive
    id_model_1 = '1GLIBTbvoY48lFUDVRULtSsX9hPi6R9oJ'  # Apel / Jeruk
    id_model_2 = '1Iwz4rct0Ao3vzkgpReJH9UK3JNpmwY4T'  # Tomat

    # Download hanya jika belum ada atau file kosong
    if not os.path.exists(path_model_1) or os.path.getsize(path_model_1) == 0:
        with st.spinner("Mendownload model Apel vs Jeruk..."):
            gdown.download(
                id=id_model_1,
                output=path_model_1,
                quiet=False
            )

    if not os.path.exists(path_model_2) or os.path.getsize(path_model_2) == 0:
        with st.spinner("Mendownload model VGG16 Tomat..."):
            gdown.download(
                id=id_model_2,
                output=path_model_2,
                quiet=False
            )

download_models()

load_custom_css()
# =============================================================================
# KONSTANTA MODEL
# =============================================================================

# ── Mode 1: Apel vs Jeruk ──────────────────────────────────────────────────
MODEL_1_PATH   = os.path.join("models", "Model_Custom_ApelJeruk.h5")
LABELS_MODE_1  = ["Apel", "Jeruk"]
SIZE_MODE_1 = (150, 150)

# ── Mode 2: Penyakit Daun Tomat ────────────────────────────────────────────
MODEL_2_PATH   = os.path.join("models", "Model_VGG16_Tomat.h5")
LABELS_MODE_2  = [
    "Bacterial Spot",           # 0 — Bercak Bakteri
    "Early Blight",             # 1 — Hawar Awal
    "Late Blight",              # 2 — Hawar Akhir
    "Leaf Mold",                # 3 — Jamur Daun
    "Septoria Leaf Spot",       # 4 — Bercak Septoria
    "Spider Mites",             # 5 — Tungau Laba-laba
    "Target Spot",              # 6 — Bercak Target
    "Tomato Yellow Leaf Curl",  # 7 — Keriting Kuning (TYLCV)
    "Tomato Mosaic Virus",      # 8 — Virus Mosaik
    "Healthy",                  # 9 — Daun Sehat
]
SIZE_MODE_2 = (224, 224)

CHIP_TOMAT = [
    "chip-red", "chip-red", "chip-red", "chip-yellow", "chip-red",
    "chip-yellow", "chip-red", "chip-red", "chip-yellow", "chip-green",
]

LABEL_ID_MODE_2 = [
    "Bercak Bakteri", "Hawar Awal", "Hawar Akhir", "Jamur Daun",
    "Bercak Septoria", "Tungau Laba-laba", "Bercak Target",
    "Keriting Kuning (TYLCV)", "Virus Mosaik", "Daun Sehat",
]

# =============================================================================
# SIDEBAR — Navigasi & Konfigurasi Modern & Elegan
# =============================================================================
with st.sidebar:
    # Header Brand dengan Ambient Glow & Pulse Dot
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon-wrapper">
            <span class="brand-icon">🔬</span>
        </div>
        <div class="brand-info">
            <h2 class="brand-title">NEURAL VISION</h2>
            <span class="brand-badge"><span class="pulse-dot"></span> AI Engine Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Section 1: Pemilihan Mode Navigasi Card-Style
    st.markdown("""
    <div class="sidebar-section-header">
        <span class="section-tag">// NAVIGASI SISTEM</span>
        <h4 class="section-title">Pilih Mode Klasifikasi</h4>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        label="mode_radio",
        options=[
            "🍎  Mode 1 — Deteksi Apel vs Jeruk",
            "🍅  Mode 2 — Deteksi Penyakit Daun Tomat",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-glow-divider"></div>', unsafe_allow_html=True)

    # Section 2: HUD Active Model Specifications Card
    st.markdown("""
    <div class="sidebar-section-header">
        <span class="section-tag">// ARSITEKTUR AKTIF</span>
        <h4 class="section-title">Spesifikasi Model</h4>
    </div>
    """, unsafe_allow_html=True)

    if "Mode 1" in mode:
        st.markdown("""
        <div class="sidebar-card">
            <span class="sidebar-card-badge badge-purple">⚡ Custom CNN · Binary</span>
            <div class="sidebar-model-chip">
                <span style="font-size:0.95rem;">📦</span>
                <code>Model_Custom_ApelJeruk.h5</code>
            </div>
            <div class="sidebar-spec-grid">
                <div class="sidebar-spec-row">
                    <span class="sidebar-spec-label">📐 Dimensi Input</span>
                    <span class="sidebar-spec-value">150 × 150 × 3</span>
                </div>
                <div class="sidebar-spec-row">
                    <span class="sidebar-spec-label">🎯 Fungsi Output</span>
                    <span class="sidebar-spec-value">Binary Sigmoid</span>
                </div>
                <div class="sidebar-spec-row">
                    <span class="sidebar-spec-label">🔬 Normalisasi</span>
                    <span class="sidebar-spec-value">[0.0, 1.0] (1/255)</span>
                </div>
                <div class="sidebar-spec-row">
                    <span class="sidebar-spec-label">🏷️ Jumlah Target</span>
                    <span class="sidebar-spec-value">2 Kelas</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Pratinjau Kategori Target Deteksi
        st.markdown("""
        <div style="margin-top: 10px;">
            <span class="section-tag">// TARGET DETEKSI</span>
            <div class="sidebar-chips-container">
                <span class="sidebar-chip">🍎 Apel (Kelas 0)</span>
                <span class="sidebar-chip">🍊 Jeruk (Kelas 1)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="sidebar-card">
            <span class="sidebar-card-badge badge-cyan">🚀 VGG16 · Transfer Learning</span>
            <div class="sidebar-model-chip">
                <span style="font-size:0.95rem;">📦</span>
                <code>Model_VGG16_Tomat.h5</code>
            </div>
            <div class="sidebar-spec-grid">
                <div class="sidebar-spec-row">
                    <span class="sidebar-spec-label">📐 Dimensi Input</span>
                    <span class="sidebar-spec-value">224 × 224 × 3</span>
                </div>
                <div class="sidebar-spec-row">
                    <span class="sidebar-spec-label">🎯 Fungsi Output</span>
                    <span class="sidebar-spec-value">10-Class Softmax</span>
                </div>
                <div class="sidebar-spec-row">
                    <span class="sidebar-spec-label">🔬 Backbone</span>
                    <span class="sidebar-spec-value">ImageNet Pretrained</span>
                </div>
                <div class="sidebar-spec-row">
                    <span class="sidebar-spec-label">🏷️ Jumlah Target</span>
                    <span class="sidebar-spec-value">10 Diagnosa</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Pratinjau Kategori Target Deteksi (10 Kelas)
        st.markdown("""
        <div style="margin-top: 10px;">
            <span class="section-tag">// TARGET DETEKSI (10 KELAS)</span>
            <div class="sidebar-chips-container">
                <span class="sidebar-chip">🦠 Bercak Bakteri</span>
                <span class="sidebar-chip">🍂 Hawar Awal</span>
                <span class="sidebar-chip">🥀 Hawar Akhir</span>
                <span class="sidebar-chip">🍄 Jamur Daun</span>
                <span class="sidebar-chip">🔍 Bercak Septoria</span>
                <span class="sidebar-chip">🕷️ Tungau Laba-laba</span>
                <span class="sidebar-chip">🎯 Bercak Target</span>
                <span class="sidebar-chip">🟡 Keriting Kuning</span>
                <span class="sidebar-chip">🧪 Virus Mosaik</span>
                <span class="sidebar-chip" style="border-color: rgba(52,211,153,0.4); color:#6ee7b7 !important;">🌱 Daun Sehat</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-glow-divider"></div>', unsafe_allow_html=True)

    # Section 3: Footer Pengembang & Proyek
    st.markdown("""
    <div class="sidebar-footer">
        <div class="sidebar-footer-name">Ali Rahman Bayanaka</div>
        <div class="sidebar-footer-badge">NRP: 2614 · Big Data & DL</div>
        <p class="sidebar-footer-sub">Week 5 — CNN Vision Classifier Studio</p>
        <p class="sidebar-footer-sub" style="font-size:0.68rem; color:#646485 !important; margin-top:4px;">TensorFlow 2.x · Streamlit · Python 3.11</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# TENTUKAN KONFIGURASI BERDASARKAN MODE
# =============================================================================
if "Mode 1" in mode:
    MODEL_PATH  = MODEL_1_PATH
    LABELS      = LABELS_MODE_1
    TARGET_SIZE = SIZE_MODE_1
    MODE_TITLE  = "Deteksi Apel vs Jeruk"
    MODE_DESC   = "Upload foto buah untuk diklasifikasikan sebagai **Apel** atau **Jeruk**."
    MODE_ICON   = "🍎"
    N_KELAS     = len(LABELS_MODE_1)
else:
    MODEL_PATH  = MODEL_2_PATH
    LABELS      = LABELS_MODE_2
    TARGET_SIZE = SIZE_MODE_2
    MODE_TITLE  = "Deteksi Penyakit Daun Tomat"
    MODE_DESC   = "Upload foto daun tomat untuk mendeteksi penyakit atau memastikan kondisi sehat."
    MODE_ICON   = "🍅"
    N_KELAS     = len(LABELS_MODE_2)

# =============================================================================
# HERO BANNER
# =============================================================================
st.markdown(f"""
<div class="hero">
    <h1>{MODE_ICON} {MODE_TITLE}</h1>
    <p>{MODE_DESC}</p>
    <small>Model: <code>{os.path.basename(MODEL_PATH)}</code>
    &nbsp;·&nbsp; Input: {TARGET_SIZE[0]}×{TARGET_SIZE[1]} px</small>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODEL
# =============================================================================
model     = None
demo_mode = False

if not os.path.exists(MODEL_PATH):
    demo_mode = True
    st.warning(f"⚠️ **Mode Demo Aktif** — File `{MODEL_PATH}` tidak ditemukan.", icon="⚠️")
else:
    with st.spinner(f"Memuat model `{os.path.basename(MODEL_PATH)}`..."):
        try:
            model = load_tf_model(MODEL_PATH)
            st.success(f"✅ Model **{os.path.basename(MODEL_PATH)}** berhasil dimuat.", icon="✅")
        except Exception as err:
            demo_mode = True
            st.error(f"❌ Gagal memuat model: `{err}`. Beralih ke Mode Demo.", icon="❌")

st.markdown("---")

# =============================================================================
# LAYOUT UTAMA: UPLOAD (kiri) | HASIL (kanan)
# =============================================================================
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📤 Upload Gambar")
    uploaded = st.file_uploader(
        label="Pilih file gambar",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{mode}",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption=uploaded.name, use_container_width=True)
        st.caption(f"Ukuran: {image.size[0]} × {image.size[1]} px | Mode: {image.mode}")
        st.markdown("")
        predict_btn = st.button("🔍 Analisis Gambar", use_container_width=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:48px 20px; margin-top:12px;">
            <div style="font-size:3rem; margin-bottom:12px;">📂</div>
            <h3>Belum Ada Gambar</h3>
            <p>Upload foto melalui area di atas untuk memulai analisis.</p>
        </div>
        """, unsafe_allow_html=True)
        predict_btn = False

with col_result:
    st.markdown("### 📊 Hasil Prediksi")

    if not uploaded:
        st.markdown("""
        <div class="card" style="text-align:center; padding:48px 20px; margin-top:12px;">
            <div style="font-size:3rem; margin-bottom:12px;">🔮</div>
            <h3>Menunggu Gambar</h3>
            <p>Upload gambar dan klik tombol <strong>"Analisis Gambar"</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

    elif predict_btn:
        with st.spinner("Menjalankan inferensi model..."):
            img_array = preprocess_image(image, TARGET_SIZE)

            if demo_mode:
                raw = np.random.rand(N_KELAS)
                probs = raw / raw.sum()
            else:
                raw_out = model.predict(img_array, verbose=0)

            if raw_out.shape[-1] == 1 or len(raw_out[0]) == 1:
                # Model Biner Sigmoid
                p1 = float(raw_out[0][0])
                
                # --- TAMBAHKAN PRINT INI UNTUK CEK DI TERMINAL ---
                print(f"DEBUG - Nilai mentah model (p1): {p1}")
                
                probs = np.array([1.0 - p1, p1])  
            else:
                probs = raw_out[0]

        idx_top   = int(np.argmax(probs))
        label_top = LABELS[idx_top]
        conf_top  = float(probs[idx_top]) * 100

        if "Mode 2" in mode:
            sub_text = LABEL_ID_MODE_2[idx_top]
        else:
            sub_text = "Apel 🍎" if label_top == "Apel" else "Jeruk 🍊"

        st.markdown(f"""
        <div class="result-box">
            <span class="result-label">{label_top}</span>
            <p style="font-size:1rem; color:#a78bfa !important; font-weight:600; margin:0 0 6px;">{sub_text}</p>
            <span class="result-conf">Confidence: <strong style="color:#a78bfa;">{conf_top:.1f}%</strong></span>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Prediksi", label_top)
        m2.metric("Confidence", f"{conf_top:.1f}%")
        m3.metric("Status", "Demo 🎭" if demo_mode else "AI Ready 🤖")

        st.markdown("---")
        st.markdown("#### 📈 Distribusi Probabilitas")

        sorted_idx = np.argsort(probs)[::-1]
        for rank, idx in enumerate(sorted_idx):
            pct = float(probs[idx]) * 100
            lbl = LABELS[idx]
            is_top = (idx == idx_top)

            if "Mode 2" in mode:
                chip_cls = CHIP_TOMAT[idx]
            else:
                chip_cls = "chip-green" if lbl == "Apel" else "chip-yellow"

            c1, c2, c3, c4 = st.columns([0.4, 2.8, 5, 1])
            with c1:
                st.markdown(f"<div style='font-size:0.85rem; color:{'#a78bfa' if is_top else '#444460'};'>{'🏆' if rank==0 else f'{rank+1}.'}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f'<span class="chip {chip_cls}" style="font-size:0.78rem;">{lbl}</span>', unsafe_allow_html=True)
            with c3:
                st.progress(pct / 100)
            with c4:
                st.markdown(f"<div style='font-size:0.85rem; font-weight:600; color:{'#a78bfa' if is_top else '#555570'};'>{pct:.1f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:40px 20px; margin-top:12px;">
            <div style="font-size:2.5rem; margin-bottom:12px;">👈</div>
            <h3>Siap Dianalisis</h3>
            <p>Klik tombol <strong>"Analisis Gambar"</strong> di sebelah kiri.</p>
        </div>
        """, unsafe_allow_html=True)