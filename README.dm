# 🧠 CIFAR-10 Görsel Sınıflandırıcı / Image Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**[🇹🇷 Türkçe](#-türkçe-dokümantasyon) · [🇬🇧 English](#-english-documentation)**

</div>

---

## 🇹🇷 Türkçe Dokümantasyon

### 📌 Proje Hakkında

Bu proje, **CIFAR-10** veri seti üzerinde eğitilmiş bir Evrişimli Sinir Ağı'nı (CNN) kullanarak görselleri 10 farklı kategoriye sınıflandıran **Streamlit** tabanlı bir web uygulamasıdır.

Model aşağıdaki sınıfları tanıyabilmektedir:

| İndeks | Etiket      | İndeks | Etiket      |
|--------|-------------|--------|-------------|
| 0      | ✈️ Airplane  | 5      | 🐶 Dog       |
| 1      | 🚗 Automobile| 6      | 🐸 Frog      |
| 2      | 🐦 Bird      | 7      | 🐴 Horse     |
| 3      | 🐱 Cat       | 8      | 🚢 Ship      |
| 4      | 🦌 Deer      | 9      | 🚛 Truck     |

---

### 🏗️ Proje Yapısı

```
cifar10-classifier/
│
├── app.py                          # Ana Streamlit uygulaması
│
├── model/
│   └── cifar10_best_model.keras    # Eğitilmiş model dosyası (Git'e ekleme!)
│
├── requirements.txt                # Python bağımlılıkları
├── .gitignore                      # Büyük dosyaları hariç tut
└── README.md                       # Bu dosya
```

> ⚠️ **Önemli:** `model/` klasörü GitHub'a yüklenmez (`.gitignore`'da). Modeli indirmek için aşağıdaki "Model İndirme" bölümüne bakın.

---

### ⚙️ Model Mimarisi

Eğitim sırasında kullanılan CNN yapısı:

```
Giriş (32×32×3)
  └── Veri Artırma (Yatay çevirme, Öteleme, Döndürme)
  └── Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.2)
  └── Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.3)
  └── Conv2D(256) → BN → Conv2D(256) → BN → MaxPool → Dropout(0.4)
  └── GlobalAveragePooling
  └── Dense(512) → BN → Dropout(0.5)
  └── Dense(10, softmax)  →  Çıkış
```

**Optimizer:** SGD (lr=0.1, momentum=0.9, Nesterov)  
**Regularizasyon:** L2 (λ=1e-4) + BatchNorm + Dropout  
**Test Doğruluğu:** ~%90  

---

### 🚀 Kurulum ve Çalıştırma

#### 1. Depoyu klonlayın

```bash
git clone https://github.com/KULLANICI_ADINIZ/cifar10-classifier.git
cd cifar10-classifier
```

#### 2. Sanal ortam oluşturun (önerilir)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

#### 3. Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

#### 4. Model dosyasını yerleştirin

Eğitilmiş modeli `model/` klasörüne koyun:

```
model/
└── cifar10_best_model.keras
```

#### 5. Uygulamayı başlatın

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` açılacaktır.

---

### 🖥️ Uygulama Özellikleri

| Özellik | Detay |
|---------|-------|
| 🌐 Dil Desteği | Türkçe / İngilizce |
| 🖼️ Tekli Yükleme | JPG / PNG görsel yükle, anında tahmin al |
| 📂 Toplu İşlem | Birden fazla görsel yükle, hepsini toplu işle |
| 📊 Görsel Karşılaştırma | Orijinal, 32×32 işlenmiş, tahmin sonucu yan yana |
| 📈 Top-3 Tahmin | En yüksek 3 olasılığı bar grafik ile göster |
| 📥 CSV İndir | Toplu tahmin sonuçlarını CSV formatında indir |

---

### 📦 Normalizasyon Detayları

Uygulamada görsellere eğitimle **aynı dönüşümler** uygulanır:

```python
# Görseli [0, 1] aralığına ölçekle
arr = np.array(image, dtype="float32") / 255.0

# CIFAR-10 kanal istatistikleriyle normalize et
MEAN = [0.4914, 0.4822, 0.4465]   # RGB kanalları için ortalama
STD  = [0.2470, 0.2435, 0.2616]   # RGB kanalları için standart sapma

arr_normalized = (arr - MEAN) / STD
```

---

### 📋 requirements.txt

```
streamlit>=1.35.0
keras>=3.0.0
tensorflow>=2.16.0
numpy>=1.24.0
pillow>=10.0.0
matplotlib>=3.7.0
pandas>=2.0.0
```

---

### 🔒 .gitignore

```gitignore
# Model dosyaları (büyük boyutlu, Git LFS kullanın)
model/
*.keras
*.h5

# Python önbelleği
__pycache__/
*.pyc
.env
venv/
*.egg-info/

# Streamlit gizli ayarlar
.streamlit/secrets.toml

# IDE
.vscode/
.idea/
```

---

### 🤝 Katkıda Bulunma

1. Bu depoyu fork edin
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m "Yeni özellik: XYZ"`)
4. Branch'i push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request açın

---

### 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

---
---

## 🇬🇧 English Documentation

### 📌 About the Project

This is a **Streamlit** web application that classifies images into 10 categories using a Convolutional Neural Network (CNN) trained on the **CIFAR-10** dataset.

The model can recognize the following classes:

| Index | Label       | Index | Label  |
|-------|-------------|-------|--------|
| 0     | ✈️ Airplane  | 5     | 🐶 Dog  |
| 1     | 🚗 Automobile| 6     | 🐸 Frog |
| 2     | 🐦 Bird      | 7     | 🐴 Horse|
| 3     | 🐱 Cat       | 8     | 🚢 Ship |
| 4     | 🦌 Deer      | 9     | 🚛 Truck|

---

### 🏗️ Project Structure

```
cifar10-classifier/
│
├── app.py                          # Main Streamlit application
│
├── model/
│   └── cifar10_best_model.keras    # Trained model (not uploaded to Git!)
│
├── requirements.txt                # Python dependencies
├── .gitignore                      # Exclude large files
└── README.md                       # This file
```

> ⚠️ **Note:** The `model/` directory is excluded from GitHub (via `.gitignore`). See "Model Download" section below.

---

### ⚙️ Model Architecture

CNN architecture used during training:

```
Input (32×32×3)
  └── Data Augmentation (Horizontal flip, Translation, Rotation)
  └── Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.2)
  └── Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.3)
  └── Conv2D(256) → BN → Conv2D(256) → BN → MaxPool → Dropout(0.4)
  └── GlobalAveragePooling
  └── Dense(512) → BN → Dropout(0.5)
  └── Dense(10, softmax)  →  Output
```

**Optimizer:** SGD (lr=0.1, momentum=0.9, Nesterov)  
**Regularization:** L2 (λ=1e-4) + BatchNorm + Dropout  
**Test Accuracy:** ~90%  

---

### 🚀 Installation & Running

#### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cifar10-classifier.git
cd cifar10-classifier
```

#### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Place the model file

Put your trained model in the `model/` directory:

```
model/
└── cifar10_best_model.keras
```

#### 5. Launch the app

```bash
streamlit run app.py
```

Your browser will automatically open `http://localhost:8501`.

---

### 🖥️ Application Features

| Feature | Detail |
|---------|--------|
| 🌐 Language | Turkish / English toggle |
| 🖼️ Single Upload | Upload JPG / PNG, get instant prediction |
| 📂 Batch Processing | Upload multiple images, process all at once |
| 📊 Visual Comparison | Original, 32×32 processed, and prediction side-by-side |
| 📈 Top-3 Predictions | Bar chart showing top 3 class probabilities |
| 📥 CSV Export | Download batch prediction results as CSV |

---

### 📦 Normalization Details

The app applies the **exact same transformations** used during training:

```python
# Scale image to [0, 1] range
arr = np.array(image, dtype="float32") / 255.0

# Normalize with CIFAR-10 channel statistics
MEAN = [0.4914, 0.4822, 0.4465]   # Per-channel mean
STD  = [0.2470, 0.2435, 0.2616]   # Per-channel std deviation

arr_normalized = (arr - MEAN) / STD
```

---

### 📋 requirements.txt

```
streamlit>=1.35.0
keras>=3.0.0
tensorflow>=2.16.0
numpy>=1.24.0
pillow>=10.0.0
matplotlib>=3.7.0
pandas>=2.0.0
```

---

### 🤝 Contributing

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m "Add new feature: XYZ"`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

### 📄 License

This project is licensed under the [MIT License](LICENSE).