import os
import io

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
from keras.models import load_model



st.set_page_config(
    page_title="CIFAR-10 Classifier",   #Tarayıcı sekmesinde görünen yazı
    page_icon="🧠",                      
    layout="wide",                       
    initial_sidebar_state="expanded",    
)


st.markdown(
    """
    <style>
    /* ── Genel arka plan rengi ─────────────────────────────── */
    .stApp {
        background-color: #0f1117;   /* Koyu lacivert-gri zemin */
        color: #e8eaf6;              /* Ana metin rengi: açık mor-beyaz */
    }
 
    /* ── Yan panel (sidebar) arka planı ───────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #1a1d2e;   /* Biraz daha koyu, belirgin ayrım */
    }
 
    /* ── Kart bileşeni: metrikler ve sonuçlar için ─────────── */
    .metric-card {
        background: linear-gradient(135deg, #1e2140 0%, #252a45 100%);
        border: 1px solid #3d4266;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 6px 0;
    }
 
    /* ── Kart başlık metni ─────────────────────────────────── */
    .metric-card h3 {
        font-size: 0.85rem;
        color: #7986cb;             /* Mor-mavi ton */
        margin: 0 0 8px 0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
 
    /* ── Kart büyük değer metni ────────────────────────────── */
    .metric-card h2 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        color: #e8eaf6;
    }
 
    /* ── Tahmin sonucu büyük banner ────────────────────────── */
    .prediction-banner {
        background: linear-gradient(135deg, #283593 0%, #1565c0 100%);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 4px 24px rgba(40, 53, 147, 0.4);
    }
    .prediction-banner h1 {
        font-size: 2.2rem;
        margin: 0;
        color: #ffffff;
    }
    .prediction-banner p {
        font-size: 1rem;
        color: #90caf9;
        margin: 4px 0 0 0;
    }
 
    /* ── Bölüm başlıkları ──────────────────────────────────── */
    .section-header {
        border-left: 4px solid #5c6bc0;
        padding-left: 12px;
        margin: 24px 0 16px 0;
        font-size: 1.2rem;
        font-weight: 600;
        color: #c5cae9;
    }
 
    /* ── Etiket listesi rozeti ─────────────────────────────── */
    .label-badge {
        display: inline-block;
        background: #1e2140;
        border: 1px solid #3d4266;
        border-radius: 20px;
        padding: 4px 14px;
        margin: 4px;
        font-size: 0.82rem;
        color: #9fa8da;
    }
 
    /* ── Toplu işlem sonuç tablosu satırı ─────────────────── */
    .batch-row {
        display: flex;
        align-items: center;
        background: #1a1d2e;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 6px 0;
        border: 1px solid #2e3256;
        gap: 12px;
    }
 
    /* ── Streamlit progress çubuğu rengi ───────────────────── */
    .stProgress > div > div > div {
        background-color: #5c6bc0;
    }
 
    /* ── Yükleme (upload) alanı ────────────────────────────── */
    [data-testid="stFileUploader"] {
        border: 2px dashed #3d4266 !important;
        border-radius: 12px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



TEXTS = {
    # ── Türkçe ──────────────────────────────────────────────────────────────
    "TR": {
        "app_title":        "🧠 CIFAR-10 Görsel Sınıflandırıcı",
        "app_subtitle":     "Derin öğrenme ile görsel tanıma",
        "language_label":   "🌐 Dil / Language",
        "model_section":    "📊 Model Performansı",
        "accuracy_label":   "Test Doğruluğu",
        "loss_label":       "Test Kaybı",
        "labels_section":   "🏷️ Sınıflandırılabilir Etiketler",
        "upload_section":   "📂 Görsel Yükleme",
        "mode_label":       "İşlem Modu",
        "mode_single":      "Tekli Görsel",
        "mode_batch":       "Toplu İşlem (Klasör)",
        "upload_single":    "Bir görsel seçin (JPG / PNG)",
        "upload_batch":     "Birden fazla görsel seçin",
        "orig_title":       "🖼️ Orijinal Görsel",
        "proc_title":       "🔬 İşlenmiş (32×32)",
        "pred_title":       "🎯 Tahmin Sonucu",
        "confidence":       "Güven Skoru",
        "top3_title":       "📈 İlk 3 Tahmin",
        "batch_results":    "📋 Toplu Sonuçlar",
        "batch_img":        "Görsel",
        "batch_pred":       "Tahmin",
        "batch_conf":       "Güven",
        "no_model":         "⚠️ Model bulunamadı! `model/cifar10_best_model.keras` yolunu kontrol edin.",
        "model_loaded":     "✅ Model başarıyla yüklendi",
        "processing":       "İşleniyor…",
        "info_preprocess":  "Eğitimde kullanılan normalizasyon (mean/std) bu görsel üzerinde de uygulandı.",
        "summary_table":    "Özet Tablo",
        "download_csv":     "📥 CSV İndir",
    },
    # ── English ─────────────────────────────────────────────────────────────
    "EN": {
        "app_title":        "🧠 CIFAR-10 Image Classifier",
        "app_subtitle":     "Visual recognition with deep learning",
        "language_label":   "🌐 Dil / Language",
        "model_section":    "📊 Model Performance",
        "accuracy_label":   "Test Accuracy",
        "loss_label":       "Test Loss",
        "labels_section":   "🏷️ Classifiable Labels",
        "upload_section":   "📂 Image Upload",
        "mode_label":       "Processing Mode",
        "mode_single":      "Single Image",
        "mode_batch":       "Batch (Multiple Images)",
        "upload_single":    "Select an image (JPG / PNG)",
        "upload_batch":     "Select multiple images",
        "orig_title":       "🖼️ Original Image",
        "proc_title":       "🔬 Processed (32×32)",
        "pred_title":       "🎯 Prediction",
        "confidence":       "Confidence Score",
        "top3_title":       "📈 Top 3 Predictions",
        "batch_results":    "📋 Batch Results",
        "batch_img":        "Image",
        "batch_pred":       "Prediction",
        "batch_conf":       "Confidence",
        "no_model":         "⚠️ Model not found! Check `model/cifar10_best_model.keras` path.",
        "model_loaded":     "✅ Model loaded successfully",
        "processing":       "Processing…",
        "info_preprocess":  "The same normalization (mean/std) used during training was applied to this image.",
        "summary_table":    "Summary Table",
        "download_csv":     "📥 Download CSV",
    },
}
 

LABEL_MAP = {
    0: "✈️ Airplane",
    1: "🚗 Automobile",
    2: "🐦 Bird",
    3: "🐱 Cat",
    4: "🦌 Deer",
    5: "🐶 Dog",
    6: "🐸 Frog",
    7: "🐴 Horse",
    8: "🚢 Ship",
    9: "🚛 Truck",
}
 

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])  
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616]) 


@st.cache_resource          # Fonksiyon çıktısını önbelleğe alma
def load_cifar_model(path: str):
    """
    Diske kaydedilen Keras modelini belleğe yükler.
 
    Parametre
    ---------
    path : str
        .keras dosyasının tam yolu.
 
    Döndürür
    --------
    model  : yüklenen Keras modeli (başarı)
    None   : dosya bulunamazsa
    """
    if not os.path.exists(path):   # Dosya yoksa None döndür, uygulama çökmez
        return None
    return load_model(path)       
 
 
def preprocess_image(pil_img: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """
    Kullanıcıdan gelen PIL görselini modele uygun NumPy dizisine çevirir.
 
    Adımlar
    -------
    1. RGB'ye zorla   → RGBA veya grayscale görseller hata vermez
    2. 32×32 yeniden boyutlandır → CIFAR-10 modeli bu boyutu bekler
    3. [0,255] → [0,1] → normalize et  (eğitimdeki transform ile aynı)
    4. Batch boyutu ekle: (32,32,3) → (1,32,32,3)  → model bu şekli bekler
 
    Döndürür
    --------
    arr_batch : shape (1,32,32,3) float32 dizi — modele verilir
    resized   : PIL Image 32×32  — arayüzde göstermek için
    """
    img_rgb    = pil_img.convert("RGB")                          
    resized    = img_rgb.resize((32, 32), Image.LANCZOS)        
    arr        = np.array(resized, dtype="float32") / 255.0      
    arr_norm   = (arr - CIFAR10_MEAN) / CIFAR10_STD              
    arr_batch  = np.expand_dims(arr_norm, axis=0)                
    return arr_batch, resized
 
 
def predict(model, arr_batch: np.ndarray) -> tuple[int, float, np.ndarray]:
    """
    Modeli çalıştırır ve sonuçları yorumlar.
 
    Döndürür
    --------
    class_idx   : int   — en yüksek olasılıklı sınıfın indeksi (0-9)
    confidence  : float — o sınıfın olasılığı (0-1 arası)
    probs       : ndarray shape (10,) — tüm sınıfların olasılıkları
    """
    probs      = model.predict(arr_batch, verbose=0)[0]  
    class_idx  = int(np.argmax(probs))                  
    confidence = float(probs[class_idx])                 
    return class_idx, confidence, probs
 
 
def plot_top3(probs: np.ndarray, t: dict) -> plt.Figure:
    """
    İlk 3 tahmini yatay bar grafik olarak çizer.
    Matplotlib figürünü Streamlit'e st.pyplot() ile verilir.
    """
    top3_idx   = np.argsort(probs)[::-1][:3]           
    top3_names = [LABEL_MAP[i] for i in top3_idx]       
    top3_vals  = [probs[i] * 100 for i in top3_idx]     
 
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("#1a1d2e")     
    ax.set_facecolor("#1a1d2e")            
 
    colors = ["#5c6bc0", "#7986cb", "#9fa8da"]  
    bars   = ax.barh(top3_names[::-1], top3_vals[::-1], color=colors[::-1])
 
    for bar, val in zip(bars, top3_vals[::-1]):
        ax.text(
            bar.get_width() + 0.5,         
            bar.get_y() + bar.get_height() / 2,  
            f"{val:.1f}%",
            va="center", ha="left",
            color="#c5cae9", fontsize=9,
        )
 
    ax.set_xlim(0, max(top3_vals) * 1.25)  
    ax.set_xlabel(t["confidence"], color="#9fa8da", fontsize=9)
    ax.tick_params(colors="#9fa8da", labelsize=9)
    for spine in ax.spines.values():       
        spine.set_visible(False)
 
    plt.tight_layout()
    return fig


with st.sidebar:
 

    lang = st.radio(
        "🌐 Dil / Language",
        options=["TR", "EN"],
        horizontal=True,      
    )
    t = TEXTS[lang]           
 
    st.markdown("---")        
 

    MODEL_PATH = os.path.join("model", "cifar10_best_model.keras")
 
    model = load_cifar_model(MODEL_PATH)   
 
    if model is None:

        st.error(t["no_model"])
        st.stop()             
    else:
        st.success(t["model_loaded"])
 
    st.markdown("---")

    st.markdown(f'<div class="section-header">{t["model_section"]}</div>', unsafe_allow_html=True)
 

    TEST_ACCURACY = 91.27  
    TEST_LOSS     = 0.4823 
 

    st.markdown(
        f"""
        <div class="metric-card">
            <h3>{t["accuracy_label"]}</h3>
            <h2>%{TEST_ACCURACY:.2f}</h2>
        </div>
        <div class="metric-card">
            <h3>{t["loss_label"]}</h3>
            <h2>{TEST_LOSS:.4f}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    st.markdown("---")
 

    st.markdown(f'<div class="section-header">{t["labels_section"]}</div>', unsafe_allow_html=True)
 
    badges = " ".join(
        f'<span class="label-badge">{name}</span>'
        for name in LABEL_MAP.values()
    )
    st.markdown(badges, unsafe_allow_html=True)


st.markdown(f"# {t['app_title']}")
st.markdown(f"*{t['app_subtitle']}*")
st.markdown("---")
 

st.markdown(f'<div class="section-header">{t["upload_section"]}</div>', unsafe_allow_html=True)
 
mode = st.radio(
    t["mode_label"],
    options=[t["mode_single"], t["mode_batch"]],
    horizontal=True,    
)


if mode == t["mode_single"]:
 
    uploaded_file = st.file_uploader(
        t["upload_single"],
        type=["jpg", "jpeg", "png"],  
        accept_multiple_files=False,   
    )
 
    if uploaded_file is not None:
 
        pil_img = Image.open(uploaded_file)
 
       
        with st.spinner(t["processing"]):    
            arr_batch, resized = preprocess_image(pil_img)
            class_idx, confidence, probs = predict(model, arr_batch)
 
        label = LABEL_MAP[class_idx]  

        st.markdown(
            f"""
            <div class="prediction-banner">
                <h1>{label}</h1>
                <p>{t["confidence"]}: <strong>{confidence*100:.1f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
 

        col1, col2, col3 = st.columns([2, 1, 2])

 
        with col1:
            st.markdown(f"**{t['orig_title']}**")
            st.image(pil_img, width='stretch')

 
        with col2:
            st.markdown(f"**{t['proc_title']}**")

            buf = io.BytesIO()
            resized.save(buf, format="PNG")    
            buf.seek(0)                         
            st.image(buf, width='stretch')
 
        with col3:
            st.markdown(f"**{t['top3_title']}**")
            fig = plot_top3(probs, t)
            st.pyplot(fig, width='stretch')
            plt.close(fig)   

        st.info(t["info_preprocess"])


else:
 
    uploaded_files = st.file_uploader(
        t["upload_batch"],
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,    # Birden fazla dosya kabul et
    )
 
    if uploaded_files:
 

        progress_bar  = st.progress(0)     
        status_text   = st.empty()         
 
        results = []    
 
        # 8.2 Dosyaları sırayla işle ─────────────────────────────────────────
        for i, file in enumerate(uploaded_files):
 
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            status_text.text(f"{t['processing']} {i+1}/{len(uploaded_files)}")
 
            pil_img = Image.open(file)
            arr_batch, resized = preprocess_image(pil_img)
            class_idx, confidence, probs = predict(model, arr_batch)
 
            results.append({
                "name":       file.name,          
                "pil":        pil_img,              
                "resized":    resized,              
                "label":      LABEL_MAP[class_idx], 
                "confidence": confidence,           
                "probs":      probs,                
            })
 
        progress_bar.empty()    
        status_text.empty()     
 
        # 8.3 Özet tablo ─────────────────────────────────────────────────────
        st.markdown(f'<div class="section-header">{t["batch_results"]}</div>', unsafe_allow_html=True)
 
        # Başlık satırı
        hcol1, hcol2, hcol3, hcol4 = st.columns([3, 2, 3, 2])
        hcol1.markdown(f"**{t['batch_img']}**")
        hcol2.markdown(f"**{t['proc_title']}**")
        hcol3.markdown(f"**{t['batch_pred']}**")
        hcol4.markdown(f"**{t['batch_conf']}**")
 
        st.markdown("---")
 
        for r in results:
            col1, col2, col3, col4 = st.columns([3, 2, 3, 2])
 
            with col1:
                st.image(r["pil"], width=80)        
 
            with col2:
                buf = io.BytesIO()
                r["resized"].save(buf, format="PNG")
                buf.seek(0)
                st.image(buf, width=64)             
 
            with col3:
                st.markdown(f"### {r['label']}")    
 
            with col4:
                pct = r["confidence"] * 100
                # st.progress 0-100 int bekler
                st.progress(int(pct))
                st.caption(f"{pct:.1f}%")           
 
        # 8.4 CSV indirme ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f'<div class="section-header">{t["summary_table"]}</div>', unsafe_allow_html=True)
 
        import pandas as pd   
 
        df = pd.DataFrame([
            {
                t["batch_img"]:  r["name"],
                t["batch_pred"]: r["label"],
                t["batch_conf"]: f"{r['confidence']*100:.1f}%",
            }
            for r in results
        ])
 
        st.dataframe(df, width='stretch')  
 

        csv_bytes = df.to_csv(index=False).encode("utf-8")
 
        st.download_button(
            label    = t["download_csv"],
            data     = csv_bytes,
            file_name= "cifar10_predictions.csv",
            mime     = "text/csv",         
        )
