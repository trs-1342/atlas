# Atlas

Atlas, bilgisayar üzerinde çalışan basit bir sesli asistan projesidir.
Şu an geliştirme aşamasındadır ve temel amacı sesli komut sisteminin altyapısını kurmaktır.

## Amaç

Bu projenin amacı:

- Sesli komutları algılayan bir sistem oluşturmak
- "hey atlas" gibi bir tetikleyici ile sistemi aktif hale getirmek
- Daha sonra komutları işleyerek aksiyon almak (program açma, arama vb.)

Şu anki versiyon sadece tetikleyici algılama üzerine odaklanmaktadır.

---

## Kurulum

### 1. Projeyi indir

git clone https://github.com/trs-1342/atlas
cd atlas

### 2. Sanal ortam oluştur

python -m venv venv
source venv/bin/activate

### 3. Gerekli paketleri kur

pip install vosk sounddevice

### 4. Model indir

wget https://alphacephei.com/vosk/models/vosk-model-small-tr-0.3.zip
unzip vosk-model-small-tr-0.3.zip

### 5. Çalıştır

python atlas.py

---

## Nasıl Çalışır

- Mikrofon sürekli dinlenir
- Konuşma metne çevrilir
- "hey atlas" algılanırsa sistem tepki verir

---

## Durum

Bu proje erken aşamadadır.

Planlanan geliştirmeler:

- Komut işleme sistemi
- Program açma
- Arama yapma
- Daha stabil wake word sistemi

---

## İletişim

GitHub: https://github.com/trs-1342
LinkedIn: https://linkedin.com/in/halilhattabh
