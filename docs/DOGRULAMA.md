# Doğrulama kapsamı

22 Kasım 2026 tarihinde yerel CPU ortamında aşağıdaki kontroller geçti:

- Notebook nbformat şeması; tüm kod hücrelerinin Python sözdizimi (Colab magic hariç).
- Yapay 400 Hz seslerde 0,1 / 2,5 / 4 / 5 saniyelik girdiler: kırpma/dolgu, 128×126×3 boyut, sonlu [0,1] değerler, eş kanallar.
- Boş ve NaN içeren seslerin reddedilmesi.
- Yapay WAV'dan iki anotasyon döngüsünün çıkarılması ve doğru etiket sırasıyla eşleşmesi.
- Baseline toplam parametre sayısı 423.748; bir train_on_batch adımı ve softmax tahmini.
- Baseline .keras kaydetme/yükleme sonrasında tahmin eşitliği; döngü tahmin fonksiyonu.
- EfficientNet yapısında bir eğitim adımı, açık katmanların varlığı ve bütün omurga BN katmanlarının donmuş kalması.
- Eğitim adımı öncesi/sonrası omurga BN moving_mean değerlerinin aynı kalması.
- Ağırlık değiştirilip birinci aşama dosyası geri yüklendiğinde tahminlerin eski hale dönmesi.
- Önceki confusion matrix sayımlarından iki modelin accuracy, macro precision/recall/F1 değerlerinin yeniden üretilmesi.

EfficientNet yapısal kontrolü `weights=None` ile yapıldı: ImageNet ağırlıkları indirilmedi.
Notebook'un gerçek çalışma ayarı `weights="imagenet"` olarak korunur.
Aşama seçiminin veri üzerindeki performansı veya callback'lerle tam fit akışı bu kısa kontrolde ölçülmedi.

**Yapılmayanlar:** Gerçek veri indirme; 6.897 döngünün yeniden işlenmesi; tam CNN/transfer öğrenme eğitimi;
Colab/T4 çalıştırması; Kaggle kimlik akışı; tüm grafik hücrelerinin uçtan uca yürütülmesi; yeni metriklerin hesaplanması.
AUC ham olasılıklar olmadan confusion matrix üzerinden doğrulanamaz; kodu incelendi, eski AUC yeniden hesaplanmadı.
Bu nedenle yeni kod için eski skorların tekrar üretildiği veya skorların iyileştiği iddia edilmez.

Kontrol ortamı:

- tensorflow-cpu: 2.21.0
- numpy: 2.3.5
- pandas: 2.2.3
- librosa: 1.0.1
- scikit-learn: 1.8.0
- nbformat: 5.11.1
