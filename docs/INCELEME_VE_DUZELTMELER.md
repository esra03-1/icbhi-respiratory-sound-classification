# Teknik inceleme ve düzeltmeler

İnceleme tarihi: 22 Kasım 2026. İncelenen kaynak commit: `08e77e721c8a583b0ce7cf012d46d08a9bdbc4b4`.
Kaynaklar: GitHub'daki notebook, README ve altı grafik; yüklenen proje yönergesi ve 7 sayfalık PDF rapor.
Notebook kodu ve kayıtlı çıktılar incelendi. Özgün 3,7 GB veri indirilerek tam eğitim tekrarlanmadı.
Düzenlenmiş notebook yeni deneydir; eski skorlar yeni kodun sonuçları değildir.

## Genel değerlendirme

Konu, ikinci hafta ödevi için yeterince kapsamlıdır: ses işleme, grup bazlı deney tasarımı, sıfırdan CNN,
transfer öğrenme ve sınıf bazlı değerlendirmeyi birlikte içerir. Şu aşamada üçüncü bir mimari eklemekten
önce deney seçimindeki ve sonuç yorumundaki sorunları çözmek daha yararlıdır.
Hasta bazlı bölme, ortak test kümesi, eğitimden hesaplanan sınıf ağırlıkları ve Macro F1 olumlu tercihlerdir.

## Yönerge ile eşleştirme

| İster | Mevcut durum | Yapılacak düzeltme |
|---|---|---|
| Girdi/hedef/ihtiyaç | Var | Hastalık tanısı ile ses örüntüsü ayrımını açık yaz |
| Veri kaynağı, büyüklük, sınıflar | Var | Ham 6.898 ile kullanılan 6.897 döngüyü ayır |
| En az üç EDA incelemesi | Var: sınıf, cihaz, süre ve örnek spektrogram | Her grafiğin altına gözlem ve etkisini ekle |
| Ön işleme gerekçeleri | Kısmen var | Dolgu/kırpma, normalizasyon, kanallar ve pencere kaybını açıkla |
| Mimari ve eğitim ayarları | Var | Havuzlama ve Dropout açıklamalarını gerçek kodla eşleştir |
| Accuracy, precision, recall, F1, confusion matrix | Var | Dummy referansı ekle; sonuçları tek metrikle yorumlama |
| Güçlü/zayıf yönler ve hata analizi | Var | Nedensel iddiaları hipotez olarak yaz |
| Rapor, grafikler ve GitHub/Colab bağlantısı | Var | PDF 7 sayfa; önerilen 3–5 sayfaya kısalt |

Yönergede 3–5 sayfa önerilir; bu bir öneridir. İki model, CRNN veya arayüz zorunluluğu yoktur.
Kod dosyası teslimi istenmiyor; rapor içinde kod/notebook bağlantısı isteniyor.

## 1. Öncelikli teknik sorun: iki aşama arasında en iyi model kayboluyor

Özgün notebook'un EfficientNet hücresinde ilk aşamada callback yok. Doğrulama kayıpları:
- İlk aşama en iyi epoch: 1, val_loss yaklaşık 1,2371.
- İlk aşama son epoch: 8, val_loss yaklaşık 1,2980.
- İkinci aşama en iyi epoch: 3, val_loss yaklaşık 1,3059.

İkinci aşama ilk aşamanın son ağırlıklarından başlıyor. EarlyStopping yalnızca ikinci aşamadaki
adayları karşılaştırıyor; ilk aşamadaki daha iyi ağırlıklar değerlendirmeye dönmüyor.
Bu, kaydedilmiş skorların matematiksel hesabını bozmaz; fakat seçilen model tanımlanan val_loss
ölçütüne göre iki aşamanın en iyi adayı değildir.

Düzeltme: Her aşamada en iyi ağırlıkları kaydet, ikinci aşamaya birincinin en iyisinden başla,
son adayları doğrulama kaybıyla karşılaştır. Test sonuçlarıyla seçim yapma.
Yeni notebook bu akışı uygular. Macro F1 amaçsa val_loss yerine önceden belirlenmiş validation
Macro F1 seçimi ayrıca denenebilir; mevcut düzeltmede iki modelin seçim ölçütü val_loss olarak korunmuştur.

## 2. EfficientNet ince ayarında BatchNormalization

Özgün kod son 30 katmanı açarken bunların içindeki BN katmanlarını da açıyor; omurga ayrıca
`training=False` ile çağrılmıyor. Küçük ve ImageNet'ten farklı dağılımlı veri üzerinde ön eğitimli
BN istatistiklerinin değişmesi kararsızlığa katkıda bulunabilir; buradaki performans düşüşünün
kesin nedeni olduğu gösterilmiş değildir.

Düzeltme: Omurgayı ayrı model olarak kur, `base_model(x, training=False)` ile çağır;
son 30 katman içinde BN katmanlarını dondur ve değişiklikten sonra tekrar compile et.
Keras rehberi: https://keras.io/guides/transfer_learning/

Girişteki `Rescaling(255.0)` hata değildir. Veri [0,1] aralığında, Keras EfficientNet girdisi
[0,255] aralığında olmalıdır; omurganın içinde kendi ölçekleme katmanı vardır.
128×126 giriş `include_top=False` için geçerlidir; sırf standart ImageNet boyutu 224 diye yeniden
boyutlandırmak zorunlu değildir. Üç aynı kanal RGB bilgisi üretmez.
Kaynak: https://keras.io/api/applications/efficientnet/efficientnet_models/

## 3. Metrikler doğru, klinik yorum fazla güçlü

Kaydedilmiş confusion matrix sayımlarından accuracy, macro precision/recall/F1 tekrar hesaplandı;
README tablosundaki dört metrik yuvarlama düzeyinde doğrulandı. AUC ham olasılıklar olmadan
matristen doğrulanamaz; notebook'taki hesap yöntemi incelendi.

| Eski deney | Accuracy | Macro F1 | Normal Recall | Wheeze Recall | Both Recall |
|---|---:|---:|---:|---:|---:|
| Her zaman Normal diyen referans | 0,5635 | 0,1802 | 1,0000 | 0,0000 | 0,0000 |
| Baseline CNN | 0,4163 | 0,2501 | 0,5170 | 0,0203 | 0,0102 |
| EfficientNet-B0 | 0,3438 | 0,3079 | 0,3453 | 0,3041 | 0,4898 |

Testte 559/992 örnek Normal'dir. Sadece Normal söylemek %56,35 accuracy sağlar fakat azınlık
sınıflarını hiç bulamaz. CNN ve EfficientNet accuracy bakımından bu referansın altında,
Macro F1 bakımından üstündedir. İki bilgiyi birlikte sunmak gerekir.

EfficientNet'te Both Precision yaklaşık %16'dır: Both dediği örneklerin büyük kısmı yanlıştır.
Normal Recall yaklaşık %34,5 olduğundan Normal örneklerin yaklaşık %65,5'i başka sınıflara gider.
Bu yüzden 'klinik açıdan çok daha güvenilir', 'başarılı tanı sistemi' gibi sonuçlar desteklenmez.
'Bu bölmede Wheeze ve Both Recall artarken Normal ve Crackle Recall azalmış; Macro F1 yükselmiştir'
ifadesi savunulabilir. 0,2501 → 0,3079 farkı +0,0578, yani yüzde ölçeğinde +5,78 puandır;
göreli artış yaklaşık %23,1'dir. Yüzde ile yüzde puanı karıştırmamak gerekir.

Tek bölme ve tek eğitimden AUC 0,6359 → 0,6448 farkının anlamlı olduğu söylenemez.
Hasta sayısı yalnızca 126'dır; 6.898 döngü bağımsız 6.898 kişi demek değildir.
İleride hasta gruplu çapraz doğrulama veya hasta düzeyinde bootstrap ile belirsizlik raporlanabilir.

## 4. Hedef hastalık değil, akustik olay

Model KOAH, zatürre veya astım tanısı üretmiyor; bir döngüde crackle/wheeze olup olmadığını tahmin ediyor.
Normal = bu iki ek sesin etiketlenmediği döngü; sağlıklı insan demek değildir.
Önerilen başlık: 'Solunum Seslerindeki Akustik Örüntülerin CNN ve EfficientNet-B0 ile Sınıflandırılması'.
Raporun erken teşhis/teletıp ifadeleri uzun vadeli olası kullanım olarak kalmalı; bu deneyin doğrulanmış
başarısı olarak sunulmamalı.
Özgün veri açıklaması: https://bhichallenge.med.auth.gr/

## 5. Hasta bölmesi iyi, açıklaması ve kontrolleri eksik

GroupShuffleSplit aynı kişinin döngülerini farklı kümelere dağıtmıyor; bu temel sızıntı önlemi doğru.
Bununla birlikte hiçbir sızıntı türü olamaz anlamında 'tam sızıntısız' iddiası fazla geniştir.
Oranlar döngü değil gruplara uygulanır. Kayıtlı dağılım:
- Eğitim: 88 hasta, 4.915 döngü (~%71,26).
- Doğrulama: 19 hasta, 990 döngü (~%14,35).
- Test: 19 hasta, 992 döngü (~%14,38).

İki aşamalı grup bölmesi sınıf oranlarını koruma garantisi vermez.
Yeni notebook kesişim assert'leri, sınıf sayıları ve bölme manifestlerini ekler.
Resmi ICBHI bölmesi kullanılmadığı için literatürdeki challenge skorlarıyla doğrudan kıyaslanmamalı.

Özgün kodda 4 saniye penceresi tüm veri süre dağılımının %90 yüzdeliğine dayandırılıyor;
bu etiket sızıntısıyla aynı şey değildir, fakat test verisi ön işleme tasarımına etki etmiştir.
Yeni notebook bölmeyi önce yapıp EDA'yı eğitimde yürütür; 4 saniye karşılaştırma için sabit ayardır.
Tüm veriyle spektrogramları önceden üretmek tek başına sızıntı değildir: mevcut min-max işlemi örnek başınadır,
veri genelinden ortalama/std öğrenmez.

## 6. Rapor ile kod arasındaki somut uyuşmazlıklar

| Yer | Sorun | Düzeltme |
|---|---|---|
| Veri sayısı | PDF 6.898 kullanılmış gibi anlatıyor; süre filtresinden sonra toplam 6.897 | Bir döngü elendiğini ve nedenini açıkla; ham/kullanılan sayıyı ayır |
| Class weights | PDF: 0,471 / 0,927 / 1,958 / 3,418 | Kayıtlı notebook: 0,473 / 0,929 / 1,876 / 3,614; yeni deneyde yeniden hesaplanacak |
| Baseline erken durdurma | PDF 13. epoch civarında en iyi ağırlığa dönüldüğünü belirsiz anlatıyor | Eğitim 13. epochta bitti, en iyi ağırlık 7. epochtan geldi |
| Baseline bloklar | Metin her blokta pooling var izlenimi veriyor | İlk üç blok MaxPooling, dördüncüden sonra GlobalAveragePooling |
| EfficientNet başlığı | Bazı açıklamalarda Dropout(0,3) atlanmış | Dense(128) sonrasında Dropout(0,3) de var |
| Örnekleme | 'Cihaz frekans farklılıklarını giderir' ifadesi | Ortak örnekleme hızı sağlar; cihaz yanıtını eşitlemez |
| Normal etiketi | Sağlıklı solunum diye sunuluyor | Ek ses anotasyonu bulunmayan döngü olarak tanımla |
| Hata nedenleri | Dolgu/gürültü kesin neden gibi yazılmış | Görsel gözlem ve deneyle sınanacak açıklamayı ayır |

## 7. Hata analizinin sınırı

Dört yüksek skorlu yanlış örnekte siyah dolgu görülmesi nedensellik kanıtı değildir.
Örnekler tüm hataları temsil etmez. README'deki '1,5 saniyenin altında' genellemesi de bu dört
görselden doğrulanmış değildir; PDF'deki yaklaşık 1,3–2,5 saniye aralığıyla tam örtüşmez.
Sesi dinlemeden alt bant enerjisinin sürtünme gürültüsü olduğunu kesin söyleyemeyiz.

Yeni notebook kimlik/süre/tahmin tablosu ve süre/cihaz bazlı tüm test hata özetlerini üretir.
Dolgu türü karşılaştırması aynı hasta bölmesinde yalnızca bu değişken değiştirilerek yapılmalıdır.
Repeat padding ve 100–2.000 Hz filtre otomatik iyileştirme değildir: tekrar yapay sınırlar,
filtre ise yararlı bileşen kaybı oluşturabilir. Eğitim/doğrulama üzerinde ayrı deneyle ölçülmelidir.

## 8. Açıklamalar yeterli mi?

Özgün notebook 13 hücreden oluşuyor: 5 Markdown, 8 kod. İlk Markdown yalnızca Colab rozeti;
veri indirme, anotasyon, EDA ve ön işleme adımlarında çoğunlukla kod yorumları var.
Model açıklamaları daha iyi; ancak ilk kez okuyan biri temel kavramları ve karar gerekçelerini
öğrenmekte zorlanabilir. Her satıra yorum eklemek yerine her işlemden önce üç soruyu yanıtla:
'Ne yapıyorum?', 'Neden yapıyorum?', 'Çıktı nasıl okunur?'

Yeni sürüm 24 hücredir: 13 Markdown, 11 kod. Eklenen açıklamalar: döngü/kayıt ayrımı,
16 kHz, 64.000 örnek, n_fft/hop_length, Mel/dB, üç aynı kanal, örnek başına min-max,
hasta sızıntısı, class_weight, callback ve metrik yorumları.
Çalıştırdıktan sonra grafik altlarına kendi gözlemlerini ekle: her grafik için 2–3 cümle yeterli.
Deney sonucu gerektiren cümleler önceden kesin başarı iddiası olarak yazılmadı.

## 9. GitHub'da yeniden çalıştırılabilirlik

Depoda kurulum, sıralı Colab çalıştırma talimatları ve ön işleme ayarları eksikti.
Model/ağırlık kaydı, eğitim geçmişi, tahmin tablosu, bölme manifesti ve yeni ses için tahmin fonksiyonu yoktu.
Bunlar yeni notebook'a eklendi. Bağımlılıklar Colab uyumluluk aralıklarıyla verildi;
bunlar test edilmiş tam sürüm kilidi değildir. Gerçek çalıştırmanın `environment.txt` çıktısı saklanmalıdır.

`.gitignore` ham veriyi, erişim dosyasını ve model ağırlıklarını dışarıda tutar.
Depoda mevcut bir gizli anahtar tespit edildiği iddia edilmiyor; bu tekrar kullanım için önlemdir.
Kendi koduna lisans seçmek ayrıca bir sahiplik kararıdır; veri ve ImageNet ağırlıkları aynı lisansa
sahipmiş gibi lisans eklenmedi.

## 10. Uygulama sırası

1. Yeni notebook'u Colab'da aç, GPU seç ve baştan sona çalıştır.
2. Hasta/sınıf sayıları, elenen döngü ve manifestleri kontrol et.
3. Seçilen EfficientNet aşamasını, yeni modellerin Macro F1/accuracy ve sınıf raporlarını kaydet.
4. Yeni grafik ve tablolarla README sonuçlarını ve PDF'yi birlikte güncelle.
5. PDF'yi 3–5 sayfada düzenle: problem/veri; EDA/ön işleme; modeller/eğitim; sonuçlar; hata/sınırlılıklar.
6. Eğitim/doğrulama üzerinde yalnız bir geliştirme dene; örneğin SpecAugment.
7. Sağlam bir karşılaştırmadan sonra CRNN ekle. Arayüz, bu aşamada bilimsel doğruluğun önüne geçmemeli.

Burada test kümesi hata analizi için incelenmiştir. Gelecek denemeleri sürekli aynı test skoruyla
ayarlamak test kümesine uyum riskini artırır; yeni nihai değerlendirme tasarımı gerekir.

## Doğrulama kapsamı

Kodun sözdizimi, notebook yapısı, metriklerin confusion matrix ile tutarlılığı ve yapay veride
çalıştırılabilen kritik fonksiyonlar kontrol edildi. Ayrıntı `DOGRULAMA.md` dosyasındadır.
Gerçek veride indirme, tam eğitim, yeni skorlar ve GPU çalışması doğrulanmadı.
