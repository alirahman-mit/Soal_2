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

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f0f1a 0%, #0a0a14 100%);
            border-right: 1px solid rgba(139, 92, 246, 0.18);
        }
        [data-testid="stSidebar"] * {
            color: #c4c4d4 !important;
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

        /* ── Radio ── */
        .stRadio label { color: #8888aa !important; }
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
