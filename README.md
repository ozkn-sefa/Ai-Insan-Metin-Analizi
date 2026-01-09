# AI ve İnsan Metin Sınıflandırma Sistemi

Bu proje, bir metnin **yapay zeka (AI)** tarafından mı yoksa **insan** tarafından mı yazıldığını tespit etmek amacıyla geliştirilmiş **bir makine öğrenmesi** sistemidir. Modern **NLP (Doğal Dil İşleme)** teknikleri ve farklı **derin öğrenme mimarileri** kullanılarak yüksek doğruluklu bir sınıflandırma hedeflenmiştir.

---

##  Proje Bileşenleri

Sistem üç ana katmandan oluşur:

###  Eğitim Modülü

* BERT tabanlı embedding vektörleri kullanılır
* Aşağıdaki modeller eğitilir:

  * MLP (Multi-Layer Perceptron)
  * 1D-CNN (Convolutional Neural Network)
  * BiLSTM (Bidirectional LSTM)
  * Logistic Regression

###  API Katmanı (Backend)

* **Flask** framework’ü ile geliştirilmiştir
* Eğitilmiş modelleri yükler
* Dış dünyadan gelen metin isteklerine tahmin döndürür

###  Kullanıcı Paneli (Frontend)

* Web tabanlı, modern ve responsive arayüz
* Kullanıcı metni girer ve sonucu anlık olarak görür

---

## Teknik Mimari

Metinlerin sayısal vektörlere dönüştürülmesi için **Sentence Transformers** kullanılır:

* Model: `all-mpnet-base-v2`
* Çıktı: **768 boyutlu embedding vektörü**

Bu vektörler aşağıdaki modeller tarafından analiz edilir:

* **MLP**: Derin sinir ağı mimarisi
* **1D-CNN**: Yerel kelime ve n-gram kalıplarını yakalar
* **BiLSTM**: Metni çift yönlü bağlamsal olarak analiz eder
* **Logistic Regression**: Temel ve karşılaştırmalı sınıflandırma modeli

---

## Kurulum ve Çalıştırma Rehberi

Projeyi çalıştırmak için adımları sırasıyla takip ediniz.

### Gerekli Kütüphanelerin Yüklenmesi

```bash
pip install pandas torch sentence-transformers scikit-learn joblib flask flask-cors openpyxl
```

---

### Modellerin Eğitilmesi

Web arayüzünü başlatmadan önce modellerin eğitilmesi gerekir.

```bash
python egitim_scripti.py
```

 **Not:**
Bu işlem tamamlandığında `modeller/` klasörü içinde aşağıdaki dosyaların oluştuğundan emin olun:

* `.pt`
* `.joblib`

---

###  Web Sunucusunun Başlatılması

```bash
python web_motoru.py
```

Sunucu çalıştıktan sonra tarayıcıdan aşağıdaki adrese gidin:

```
http://localhost:5000
```

---

## Dosya Yapısı

```
├── mleğitim.py
├── webmotor.py
├── templates/
│   └── index.html
├── modeller/
├── dökumanlar/
├── model_karsilastirma_sonuclari.xlsx
└── Readme.md
```

### Dosyaların Görevleri

* **mleğitim.py**

  * Veri setini dengeler
  * Modelleri eğitir
  * Performans raporu oluşturur

* **webmotor.py**

  * Flask tabanlı API sunucusu
  * Modelleri yükler ve tahmin üretir

* **templates/index.html**

  * Kullanıcı arayüzü (UI)

* **modeller/**

  * Eğitilmiş model ağırlıkları ve konfigürasyonlar

* **dökumanlar/**

  * Eğitimde kullanılan Excel veri setleri

---

##  Performans Takibi

Eğitim tamamlandıktan sonra proje dizininde oluşan:

**`model_karsilastirma_sonuclari.xlsx`**

Dosyası üzerinden:

* Modellerin doğruluk oranlarını
* Eğitim sürelerini
* Karşılaştırmalı performans analizlerini

inceleyebilirsiniz.

---

## Özet

Bu proje; **NLP**, **derin öğrenme**, **backend–frontend entegrasyonu** ve **model karşılaştırma** süreçlerini kapsayan, akademik ve endüstriyel kullanıma uygun kapsamlı bir AI uygulamasıdır.
