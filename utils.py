# =============================================================================
# utils.py — Pusat Logika Aplikasi Deteksi Gambar CNN
# Berisi: CSS kustom dark mode, loader model TF/Keras, dan preprocessor gambar.
# =============================================================================

import numpy as np
import streamlit as st


# ===========================================================================
# 1. CSS KUSTOM — Dark Mode Modern + Font Inter
# ===========================================================================

def load_custom_css():
    """
    Memuat CSS kustom untuk seluruh halaman aplikasi.
    Menerapkan tema dark mode profesional, font Inter, efek glassmorphism,
    animasi hover, dan komponen kartu kustom.
    """
    css = """
    <style>
        /* ── Import Font Inter ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ── Base ── */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }

        /* ── Background utama ── */
        .stApp {
            background: #090912;
        }

        /* ── Sidebar Container & Aesthetics ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0e0d1d 0%, #090815 45%, #06050e 100%) !important;
            border-right: 1px solid rgba(139, 92, 246, 0.16) !important;
            box-shadow: 6px 0 35px rgba(0, 0, 0, 0.6) !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-top: 1.4rem !important;
            padding-bottom: 2rem !important;
            padding-left: 1.1rem !important;
            padding-right: 1.1rem !important;
        }
        
        /* Custom Scrollbar for Sidebar */
        [data-testid="stSidebar"]::-webkit-scrollbar {
            width: 5px;
        }
        [data-testid="stSidebar"]::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.2);
        }
        [data-testid="stSidebar"]::-webkit-scrollbar-thumb {
            background: rgba(139, 92, 246, 0.25);
            border-radius: 999px;
        }
        [data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
            background: rgba(139, 92, 246, 0.55);
        }

        /* Scoped typography in sidebar */
        [data-testid="stSidebar"] p {
            color: #9494b0;
            font-size: 0.88rem;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {
            color: #f1f1f9 !important;
            font-weight: 700;
        }

        /* ── Tipografi ── */
        h1 { color: #f0f0ff !important; font-weight: 800 !important; letter-spacing: -0.5px; }
        h2 { color: #e2e2f0 !important; font-weight: 700 !important; }
        h3 { color: #c4c4d4 !important; font-weight: 600 !important; }
        p, li  { color: #8888aa !important; line-height: 1.7; }
        label  { color: #8888aa !important; font-weight: 500 !important; }
        small  { color: #555570 !important; }

        /* ── Hero / Banner halaman ── */
        .hero {
            background: linear-gradient(135deg, #1a1040 0%, #0d0d1e 60%, #1a1040 100%);
            border: 1px solid rgba(139, 92, 246, 0.25);
            border-radius: 20px;
            padding: 36px 40px;
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
        }
        .hero::before {
            content: '';
            position: absolute;
            top: -60px; right: -60px;
            width: 260px; height: 260px;
            background: radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .hero h1  { font-size: 1.9rem; margin: 0 0 6px; }
        .hero p   { color: rgba(180,180,210,0.85) !important; margin: 0; font-size: 0.97rem; }
        .hero small { color: rgba(120,120,160,0.7) !important; font-size: 0.82rem; }

        /* ── Kartu info ── */
        .card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 16px;
            padding: 20px 22px;
            margin: 6px 0;
            transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
        }
        .card:hover {
            border-color: rgba(139,92,246,0.35);
            transform: translateY(-2px);
            box-shadow: 0 12px 36px rgba(0,0,0,0.45);
        }
        .card h3 {
            font-size: 0.95rem;
            background: linear-gradient(90deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        }
        .card p { font-size: 0.85rem; color: #555570 !important; }

        /* ── Kotak hasil prediksi ── */
        .result-box {
            background: linear-gradient(135deg, rgba(139,92,246,0.1), rgba(96,165,250,0.07));
            border: 1px solid rgba(139,92,246,0.28);
            border-radius: 18px;
            padding: 30px 24px;
            text-align: center;
            margin: 16px 0;
        }
        .result-label {
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: block;
            line-height: 1.2;
            margin-bottom: 8px;
        }
        .result-conf {
            font-size: 0.88rem;
            color: #555570 !important;
        }

        /* ── Badge chip ── */
        .chip {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin: 3px 2px;
        }
        .chip-purple { background: rgba(139,92,246,0.18); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.35); }
        .chip-blue   { background: rgba(96,165,250,0.18);  color: #93c5fd; border: 1px solid rgba(96,165,250,0.35); }
        .chip-green  { background: rgba(52,211,153,0.18);  color: #6ee7b7; border: 1px solid rgba(52,211,153,0.35); }
        .chip-red    { background: rgba(248,113,113,0.18); color: #fca5a5; border: 1px solid rgba(248,113,113,0.35); }
        .chip-yellow { background: rgba(251,191,36,0.18);  color: #fde68a; border: 1px solid rgba(251,191,36,0.35); }

        /* ── Tombol utama ── */
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 13px 28px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.3px !important;
            transition: opacity 0.2s, transform 0.2s !important;
        }
        .stButton > button:hover {
            opacity: 0.87 !important;
            transform: translateY(-1px) !important;
        }

        /* ── File uploader ── */
        [data-testid="stFileUploadDropzone"] {
            background: rgba(139,92,246,0.04) !important;
            border: 2px dashed rgba(139,92,246,0.3) !important;
            border-radius: 14px !important;
            transition: border-color 0.2s !important;
        }
        [data-testid="stFileUploadDropzone"]:hover {
            border-color: rgba(139,92,246,0.6) !important;
        }

        /* ── Metrik ── */
        [data-testid="stMetricValue"] {
            color: #a78bfa !important;
            font-weight: 800 !important;
            font-size: 2rem !important;
        }
        [data-testid="stMetricLabel"] { color: #555570 !important; }

        /* ── Progress bar ── */
        [data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
            border-radius: 999px !important;
        }

        /* ── Gambar ── */
        [data-testid="stImage"] img {
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.07);
        }

        /* ── Divider ── */
        hr { border-color: rgba(255,255,255,0.06) !important; }

        /* ── Alert/info box ── */
        [data-testid="stAlert"] { border-radius: 12px !important; }

        /* ── Radio bawaan (General) ── */
        .stRadio label { color: #8888aa !important; }

        /* ── SIDEBAR ELEGAN & MODERN KOMPONEN ── */

        /* 1. Header Brand Sidebar */
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 18px 18px;
            background: linear-gradient(135deg, rgba(124, 58, 237, 0.15) 0%, rgba(37, 99, 235, 0.08) 100%);
            border: 1px solid rgba(139, 92, 246, 0.28);
            border-radius: 18px;
            margin-bottom: 22px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }
        .sidebar-brand::before {
            content: '';
            position: absolute;
            top: -40px;
            right: -40px;
            width: 110px;
            height: 110px;
            background: radial-gradient(circle, rgba(167, 139, 250, 0.22) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .brand-icon-wrapper {
            position: relative;
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.45rem;
            box-shadow: 0 0 20px rgba(124, 58, 237, 0.45);
            flex-shrink: 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .brand-info {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .brand-title {
            font-size: 1.08rem !important;
            font-weight: 800 !important;
            margin: 0 !important;
            letter-spacing: 0.6px;
            background: linear-gradient(90deg, #ffffff 0%, #c4b5fd 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.2 !important;
        }
        .brand-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.7rem;
            font-weight: 700;
            color: #34d399 !important;
            letter-spacing: 0.8px;
            text-transform: uppercase;
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10b981;
            animation: pulse-glow 2s infinite ease-in-out;
            display: inline-block;
        }
        @keyframes pulse-glow {
            0%   { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.75); }
            70%  { transform: scale(1.15); box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        /* 2. Section Header & Glow Dividers */
        .sidebar-section-header {
            margin-top: 14px;
            margin-bottom: 12px;
        }
        .section-tag {
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 1.4px;
            text-transform: uppercase;
            color: #a78bfa !important;
            display: block;
            margin-bottom: 4px;
        }
        .section-title {
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            color: #f1f1fc !important;
            margin: 0 !important;
        }
        .sidebar-glow-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.35), transparent);
            margin: 20px 0;
            border: none;
        }

        /* 3. Streamlit Radio Card Styling in Sidebar */
        [data-testid="stSidebar"] div[role="radiogroup"] {
            display: flex !important;
            flex-direction: column !important;
            gap: 10px !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 14px !important;
            padding: 13px 15px !important;
            margin: 0 !important;
            cursor: pointer !important;
            transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1) !important;
            display: flex !important;
            align-items: center !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            background: rgba(139, 92, 246, 0.09) !important;
            border-color: rgba(139, 92, 246, 0.45) !important;
            transform: translateX(4px);
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.2) !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background: linear-gradient(135deg, rgba(124, 58, 237, 0.2) 0%, rgba(59, 130, 246, 0.12) 100%) !important;
            border-color: rgba(167, 139, 250, 0.65) !important;
            box-shadow: 0 0 22px rgba(124, 58, 237, 0.26), inset 0 0 14px rgba(139, 92, 246, 0.1) !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label p {
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            color: #f1f1fc !important;
            margin: 0 !important;
        }

        /* 4. Glassmorphic HUD Model Specification Card */
        .sidebar-card {
            background: linear-gradient(145deg, rgba(23, 20, 44, 0.72) 0%, rgba(13, 12, 26, 0.82) 100%);
            border: 1px solid rgba(139, 92, 246, 0.24);
            border-radius: 16px;
            padding: 18px;
            margin: 12px 0;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.38);
            position: relative;
            overflow: hidden;
        }
        .sidebar-card::before {
            content: '';
            position: absolute;
            top: -40px;
            right: -40px;
            width: 100px;
            height: 100px;
            background: radial-gradient(circle, rgba(139, 92, 246, 0.18) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .sidebar-card-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 11px;
            border-radius: 8px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 12px;
        }
        .badge-purple {
            background: rgba(139, 92, 246, 0.22);
            color: #d8b4fe !important;
            border: 1px solid rgba(139, 92, 246, 0.4);
        }
        .badge-cyan {
            background: rgba(56, 189, 248, 0.18);
            color: #7dd3fc !important;
            border: 1px solid rgba(56, 189, 248, 0.4);
        }
        .sidebar-model-chip {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(139, 92, 246, 0.2);
            border-radius: 10px;
            padding: 8px 12px;
            margin-bottom: 14px;
        }
        .sidebar-model-chip code {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            color: #e0e7ff !important;
            font-size: 0.8rem;
            background: transparent !important;
            padding: 0 !important;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .sidebar-spec-grid {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .sidebar-spec-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: rgba(255, 255, 255, 0.025);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.82rem;
        }
        .sidebar-spec-label {
            color: #8f8fae !important;
            display: flex;
            align-items: center;
            gap: 7px;
            font-size: 0.8rem;
        }
        .sidebar-spec-value {
            color: #f1f1fc !important;
            font-weight: 600;
            font-size: 0.8rem;
            background: rgba(139, 92, 246, 0.12);
            padding: 2px 8px;
            border-radius: 6px;
            border: 1px solid rgba(139, 92, 246, 0.25);
        }

        /* 5. Target Classes Tag Cloud */
        .sidebar-chips-container {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 10px;
        }
        .sidebar-chip {
            font-size: 0.74rem;
            padding: 4px 10px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.07);
            color: #b8b8d4 !important;
            transition: all 0.22s ease;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .sidebar-chip:hover {
            border-color: rgba(139, 92, 246, 0.45);
            background: rgba(139, 92, 246, 0.15);
            color: #ffffff !important;
            transform: translateY(-1px);
        }

        /* 6. Footer Card */
        .sidebar-footer {
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.02) 0%, rgba(255, 255, 255, 0.01) 100%);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            padding: 14px 16px;
            margin-top: 18px;
            text-align: center;
            position: relative;
        }
        .sidebar-footer-name {
            font-size: 0.85rem;
            font-weight: 700;
            color: #f1f1fc !important;
            margin: 0 0 2px 0;
            letter-spacing: 0.3px;
        }
        .sidebar-footer-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            background: rgba(139, 92, 246, 0.18);
            border: 1px solid rgba(139, 92, 246, 0.35);
            color: #c4b5fd !important;
            font-size: 0.7rem;
            font-weight: 600;
            margin-top: 4px;
            margin-bottom: 6px;
        }
        .sidebar-footer-sub {
            font-size: 0.72rem;
            color: #72728f !important;
            margin: 0;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ===========================================================================
# 2. LOADER MODEL TF/KERAS
# ===========================================================================

@st.cache_resource(show_spinner=False)
def load_tf_model(path: str):
    """
    Memuat model TensorFlow/Keras dari file .h5 dengan caching sesi.

    Menggunakan @st.cache_resource sehingga model hanya dimuat SATU KALI
    per sesi Streamlit, menghemat waktu dan memori secara signifikan.

    Args:
        path (str): Path lengkap ke file model .h5

    Returns:
        tf.keras.Model: Model Keras siap inferensi.

    Raises:
        FileNotFoundError : Jika file tidak ditemukan.
        Exception         : Untuk error lain saat load_model.
    """
    import tensorflow as tf
    model = tf.keras.models.load_model(path)
    return model


# ===========================================================================
# 3. PREPROCESSING GAMBAR
# ===========================================================================

def preprocess_image(image, target_size: tuple) -> np.ndarray:
    """
    Melakukan preprocessing gambar PIL agar siap dimasukkan ke model CNN.

    Alur preprocessing:
      1. Konversi ke RGB (3 channel) — menghapus channel alpha jika ada.
      2. Resize ke ukuran yang diharapkan model (width x height).
      3. Konversi ke numpy array dengan dtype float32.
      4. Normalisasi piksel: [0, 255] → [0.0, 1.0] (dibagi 255.0).
      5. Tambah dimensi batch (expand_dims): (H,W,C) → (1,H,W,C).

    Args:
        image       : Objek PIL.Image yang sudah dibuka.
        target_size : Tuple (width, height) ukuran input model.

    Returns:
        np.ndarray: Array shape (1, height, width, 3) siap untuk model.predict().
    """
    # Langkah 1: Pastikan mode RGB
    image = image.convert("RGB")

    # Langkah 2: Resize ke ukuran target model
    image = image.resize(target_size)

    # Langkah 3 & 4: Array float32 + normalisasi
    img_array = np.array(image, dtype=np.float32)

    # Langkah 5: Tambah dimensi batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
