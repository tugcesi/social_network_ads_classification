# Social Network Ads Classification – Streamlit App

Sosyal medya reklamlarına tıklayan kullanıcıların ürün satın alıp almayacağını tahmin eden interaktif bir web uygulaması.

## 📋 İçerik

| Dosya | Açıklama |
|---|---|
| `app.py` | Ana Streamlit uygulaması (4 sayfa) |
| `utils.py` | Feature engineering ve tahmin yardımcı fonksiyonları |
| `config.py` | Sabit değerler ve konfigürasyon |
| `requirements.txt` | Python bağımlılıkları |
| `Purchased.pkl` | Eğitilmiş model (model + feature listesi) |
| `Purchased.joblib` | Eğitilmiş model (yedek) |
| `Social_Network_Ads.csv` | Veri seti |

## 🚀 Kurulum & Çalıştırma

### 1. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### 2. Uygulamayı Başlatın

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` adresi açılacaktır.

## 📄 Uygulama Sayfaları

### 🏠 Ana Sayfa
- Toplam kullanıcı sayısı, satın alma istatistikleri
- Model doğruluk metrikleri
- Sınıf dağılımı ve cinsiyet bazlı satın alma oranları

### 🔮 Tahmin
- Cinsiyet, yaş ve maaş giriş formu
- Gerçek zamanlı satın alma tahmini
- Güven skoru ve olasılık görselleştirmesi

### 📊 Veri Analizi
- Özellik dağılımları (satın alma durumuna göre)
- Korelasyon matrisi
- Yaş grubu ve maaş seviyesi analizi
- Demografik scatter plot

### 🤖 Model Bilgisi
- BernoulliNB model parametreleri
- Performans metrikleri (Accuracy, Precision, Recall, F1)
- Confusion matrix
- Özellik ayrımcılık gücü (feature log-probabilities)

## 🧠 Feature Engineering

Modelin kullandığı özellikler (notebook ile aynı pipeline):

| Özellik | Açıklama |
|---|---|
| `Gender` | Cinsiyet (Female=0, Male=1) |
| `Age` | Yaş |
| `EstimatedSalary` | Tahmini maaş |
| `IsYoungRich` | Yaş < 30 ve Maaş > 50.000 ise 1 |
| `SalaryLevel_2/3/4` | Maaş seviyeleri (one-hot, drop_first=True) |
| `AgeGroup_2/3/4` | Yaş grupları (one-hot, drop_first=True) |

## 📦 Model

- **Algoritma**: BernoulliNB (Bernoulli Naive Bayes)
- **Test Doğruluğu**: ~%93
- **Eğitim/Test Ayrımı**: %80/%20 (stratified, random_state=42)
