# ARGUS AI deney protokolü

**Sürüm:** 5.0

**Durum:** Veri hazırlama, model geliştirme, GraphSAGE karşılaştırması ve tek seferlik
final test değerlendirmesi tamamlandı.

**Bilimsel referans:** `v1.0-scientific-final`
(`ce182474cd8ee36bcab2449a61347eb9803451ae`)

Bu protokol, ARGUS sonuçlarının aynı veri, zaman sınırları ve değerlendirme kuralları
altında karşılaştırılmasını sağlar.

## 1. Yeniden üretim birimi

Her çalışma aşağıdaki bilgileri kaydeder:

- veri dosyalarının SHA-256 değerleri;
- çözümlenmiş YAML yapılandırması ve rastgelelik tohumu;
- örnekleme kapsamı;
- başlangıç/bitiş zamanı ve çalışma süresi;
- kronolojik bölüm sınırları ve satır sayıları;
- üretilen dosyaların envanteri ve hash değerleri;
- doğrulama sonucu ve tamamlanmayan adımların hata kaydı.

Çalışma manifestleri `artifacts/<run>/run_manifest.json` altında tutulur. Ham veri,
ara tablolar, tahminler ve model dosyaları Git dışında kalır.

## 2. Sabit veri sürümü

| Girdi | SHA-256 | Satır |
| --- | --- | ---: |
| `data/raw/HI-Small_Trans.csv` | `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` | 5.078.345 |
| `data/raw/HI-Small_accounts.csv` | `786808526e33cfc441212dd6fccda7edfc24172149bed59c6ef59b186836b014` | 518.581 |

Farklı hash değerine sahip dosyalar ayrı bir veri sürümü olarak değerlendirilir.

## 3. Kronolojik bölümleme

İşlemler zaman damgasına göre sıralanır. Aynı dakikadaki işlemler iki farklı bölüme
ayrılmaz.

| Bölüm | Zaman aralığı | Satır | Pozitif | Pozitif oran |
| --- | --- | ---: | ---: | ---: |
| Eğitim | 2022-09-01 00:00 – 2022-09-07 14:55 | 3.554.957 | 2.856 | 0,0008033852 |
| Doğrulama | 2022-09-07 14:56 – 2022-09-09 03:16 | 761.749 | 760 | 0,0009977040 |
| Final test | 2022-09-09 03:17 – 2022-09-18 16:18 | 761.639 | 1.561 | 0,0020495274 |

İşlem kimlikleri bölümler arasında kesişmez ve sınırlar kesin olarak artar.

## 4. Ön işleme ve sızıntı kontrolü

- Medyan, ölçek, kategori sözlüğü ve frekans haritası yalnız eğitim bölümüne fit
  edilir.
- Doğrulama ve test bölümleri aynı donmuş dönüşümlerle transform edilir.
- Hedef, zaman damgası, işlem kimliği ve hesap kimliği doğrudan model girdisi olmaz.
- t anındaki geçmiş özellikleri yalnız t'den önce gerçekleşen işlemleri kullanır.
- Aynı zaman damgasındaki işlemler birbirinin geçmişine dahil edilmez.
- Banka ve hesap birleşimi `normalized_bank_id::account_id` biçiminde tekil düğüm
  kimliği oluşturur.
- Tekrarlı transferler yönlü çoklu kenar olarak korunur.
- Para tutarları farklı para birimleri arasında doğrudan toplanmaz veya
  karşılaştırılmaz.
- IBM hedefi işlem düzeyinde tutulur; hesap düzeyinde suç etiketi türetilmez.

## 5. Özellik aileleri

Kontrollü karşılaştırma üç katmanda yapılır:

1. **İşlem:** tutar, ödeme biçimi, para birimi uyumu ve işlem kategorileri.
2. **Zaman/geçmiş:** saat/gün, önceki işlem sayısı, hacim, karşı taraf çeşitliliği,
   son işlemden geçen süre ve patlama sinyalleri.
3. **Graf:** önceki fan-in, fan-out ve aynı gönderen–alıcı çiftinin tekrar sayısı.

Graf ablasyonunda model ailesi, hiperparametreler, tohum, bölümleme ve metrik kodu
sabit tutulur; yalnız özellik ailesi değişir.

## 6. Model geliştirme

### İşlem tabanlı modeller

Logistic Regression, Random Forest ve LightGBM aynı eğitim/doğrulama protokolünde
karşılaştırılır. Sınıf dengesizliği model ağırlıklarıyla ele alınır; accuracy ana
başarı ölçütü değildir.

### Model iyileştirme

Hiperparametre adayları, dış eğitim bölümünün içinde kalan üç genişleyen zamansal
fold üzerinde seçilir. Her fold kendi eğitim önekine fit edilir. Dış doğrulama,
aday seçimi tamamlandıktan sonra bir kez kullanılır.

Logistic Regression yakınsama kontrolü `n_iter < max_iter` koşuluyla yapılır.
LightGBM ve Logistic Regression için ham karar skorları sıralamada kullanılır;
olasılık doygunluğu ayrıca raporlanır.

### GraphSAGE

GraphSAGE, gönderen ve alıcı düğüm embedding'lerini işlem özellikleriyle birleştiren
kenar/işlem sınıflandırıcısıdır. Eğitim grafı yalnız geçmiş eğitim işlemlerinden
oluşur. Donanım sınırı nedeniyle deterministik örnekleme kullanılır:

- mesaj grafı: 150.000 / 1.422.288 uygun eğitim kenarı;
- denetimli eğitim: 2.396 pozitif + 50.000 deterministik negatif;
- doğrulama mesaj grafı: 300.000 / 3.554.957 eğitim kenarı;
- doğrulama skorlama: 761.749 işlemin tamamı.

Bu kapsam tam-graf GraphSAGE eğitimi olarak yorumlanmaz.

## 7. Metrikler, eşik ve skor bağları

Ana metrik average precision biçimindeki **PR-AUC**'dir. ROC-AUC ikincil metriktir.
Precision, Recall, F1, FPR, alarm sayısı, Recall@K ve Precision@K birlikte raporlanır.

Operasyonel eşik yalnız doğrulama bölümünde seçilir. Birincil kural, en fazla 5.000
alarm ve `FPR ≤ 0,01` altında Recall değerini en yüksek yapan tam skor grubunu seçer.
Eşit skorlu bir grup sırf bütçeye uyması için bölünmez.

Top-K sırası skor azalan, ardından `source_row_number` artan biçimdedir. K sınırı
bir skor bağına denk gelirse bağ içindeki örnek sırası model üstünlüğü kanıtı olarak
yorumlanmaz.

## 8. Graf katkısı ve doğrulama karşılaştırması

| Deney | Doğrulama PR-AUC |
| --- | ---: |
| Yalnız işlem | 0,09984124 |
| İşlem + zaman/geçmiş | 0,35535042 |
| İşlem + zaman/geçmiş + graf | **0,47175420** |

Aynı LightGBM protokolünde graf ailesinin eklenmesi son iki kol arasında
`+0,11640378` PR-AUC farkı üretmiştir.

| Model | Doğrulama PR-AUC | ROC-AUC |
| --- | ---: | ---: |
| Graph-enhanced LightGBM | **0,47175420** | 0,98635944 |
| Refined transaction LightGBM | 0,35535042 | 0,98179242 |
| GraphSAGE | 0,00943876 | 0,84664170 |

GraphSAGE araştırma karşılaştırmasıdır; operasyonel model graph-enhanced
LightGBM'dir.

## 9. Tek seferlik final değerlendirme

Final test açılmadan önce aşağıdaki bileşenler dondurulmuştur:

- model: `graph_enhanced_lightgbm`;
- özellik ailesi ve train-fit preprocessor;
- hiperparametreler ve rastgelelik tohumu;
- doğrulamada seçilen ham-skor eşiği;
- karşılaştırma modelleri ve metrik tanımları.

Final test tam olarak bir kez skorlanmıştır. Sonuçlara bakılarak yeniden eğitim,
özellik değişikliği, eşik değişikliği veya model seçimi yapılmamıştır.

| Rol | Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alarm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final model | Graph-enhanced LightGBM | **0,69005906** | 0,99127213 | 0,17816018 | 0,88212684 | 0,29644779 | 0,00835704 | 7.729 |
| Karşılaştırma | Refined transaction LightGBM | 0,53079643 | 0,98731443 | 0,16072332 | 0,81422165 | 0,26845496 | 0,00873200 | 7.908 |
| Araştırma karşılaştırması | GraphSAGE | 0,01341710 | 0,82662724 | 0,00600590 | 0,11338885 | 0,01140758 | 0,03854078 | 29.471 |

Test kimlik denetimi bölüm kesişimi olmadığını, satır sayısını ve tahmin
benzersizliğini doğrular. Doğrulama/test prevalans oranı `2,05424401` olarak ayrıca
raporlanır.

## 10. Artifact sözleşmesi

Başlıca yerel çıktı dizinleri:

```text
artifacts/quick/    Hızlı veri ve EDA kontrolü
artifacts/full/     Tam veri hazırlama, özellikler ve bölümleme
artifacts/sprint2/  İşlem tabanlı modeller
artifacts/sprint3/  Temporal CV, ablasyon ve eşik analizi
artifacts/sprint4/  GraphSAGE, vaka ve açıklama çıktıları
artifacts/sprint5/  Donmuş final test tahmin ve metrikleri
```

Her dizindeki manifest, çıktı hash'lerini ve kapsamı kaydeder. Final özetleri
[`../reports/generated/FINAL_EVALUATION_STATUS.md`](../reports/generated/FINAL_EVALUATION_STATUS.md)
ve
[`../reports/generated/FINAL_MODEL_COMPARISON.md`](../reports/generated/FINAL_MODEL_COMPARISON.md)
dosyalarındadır.

## 11. Çalıştırma ve doğrulama

Veri pipeline'ları:

```powershell
.\.venv\Scripts\python.exe scripts/run_quick_pipeline.py
.\.venv\Scripts\python.exe scripts/verify_run.py
.\.venv\Scripts\python.exe scripts/run_full_pipeline.py
```

Model geliştirme adımları:

```powershell
.\.venv\Scripts\python.exe scripts/train_baselines.py --config configs/baseline.yaml
.\.venv\Scripts\python.exe scripts/train_refined_models.py --config configs/refinement.yaml
.\.venv\Scripts\python.exe scripts/train_graphsage_product.py --config configs/sprint4.yaml
```

Final test yeniden çalıştırılmaz. Kayıtlı final çıktılarının salt-okunur kontrolü:

```powershell
.\.venv\Scripts\python.exe scripts/verify_final_evaluation.py --config configs/final_evaluation.yaml
```

Kod kalite kontrolleri:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check --no-cache src scripts tests app.py
.\.venv\Scripts\python.exe -m ruff format --no-cache --check src scripts tests app.py
.\.venv\Scripts\python.exe -m pip check
```

## 12. Karar sınırı

ARGUS sentetik işlem etiketleri üzerinde sıralama ve sınıflandırma performansını
ölçer. Skorlar suç isnadı veya otomatik olumsuz işlem için kullanılamaz. Gözlenen
kanıt model açıklamasından ayrı tutulur ve her vaka eğitimli bir finansal suç
analistinin değerlendirmesine bırakılır.
