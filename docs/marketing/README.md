# ARGUS Public B2B Deneyimi

Bu belge, ARGUS'un herkese açık B2B pazarlama deneyimi için mesajlaşma, kullanıcı yolculuğu ve
kanıt sınırlarını tanımlar. Bilimsel sonuçların kaynağı değildir; metrikler için
[`MODEL_CARD.md`](../MODEL_CARD.md) ile [`reports/generated/`](../../reports/generated/)
altındaki dondurulmuş raporlar esas alınır.

## Marketing Deliverables

- [ARGUS Marketing Strategy Report](ARGUS_Marketing_Strategy_Report.pdf)
- [ARGUS Marketing Strategy Portfolio](ARGUS_Marketing_Strategy_Portfolio.pdf)
- [Root Project README](../../README.md)

<p align="center">
  <a href="ARGUS_Marketing_Strategy_Portfolio.pdf">
    <img src="portfolio/portfolio-01-cover.png" alt="ARGUS Marketing Strategy Portfolio konsept kapağı" width="820">
  </a>
</p>

<p align="center">
  <img src="portfolio/portfolio-02-strategy-backbone.png" alt="ARGUS için tasarlanan altı aşamalı B2B pazarlama stratejisi" width="920">
</p>

> [!NOTE]
> Rapor ve portfolyo, ARGUS için **tasarlanmış/önerilen** B2B pazarlama stratejisini sunar.
> Konsept görseller; gerçek müşteri, kanal performansı, etkinlik, üretim kurulumu veya bilimsel
> sonuç kanıtı değildir. Görsellerdeki örnek sayılar ve senaryolar doğrulanmış proje metriği
> olarak kullanılmaz.

## Konumlandırma

**Kategori:** Graph-Based Financial Crime & Account Network Intelligence  
**Değer önerisi:** See beyond the transaction.  
**Temel konumlandırma:** ARGUS, finansal suç analistlerinin işlem geçmişini davranışsal sinyaller
ve hesap ağı bağlamıyla birleştirerek şüpheli vakaları önceliklendirmesine yardımcı olur.

ARGUS bir karar destek deneyimidir. Bir müşteriyi suçlu ilan etmez; hesabı otomatik olarak bloke
etmez, dondurmaz, yaptırım uygulamaz veya düzenleyici bildirim başlatmaz. Nihai karar, kaynak
kayıtları doğrulayan eğitimli analiste aittir.

## Hedef B2B kitle

- AML ve Financial Crime yöneticileri
- Fraud Operations yöneticileri
- Compliance direktörleri ve MLRO'lar
- Banka, dijital banka, ödeme ve elektronik para kuruluşlarındaki Data & AI ekipleri
- Finansal suç inceleme süreçlerini değerlendiren fintech ekipleri

Birincil arayüz metni bu rollere hitap eder. Samsung Innovation Campus ve capstone bilgisi depo
bağlamında korunur; ticari açılış deneyiminin ana mesajı değildir.

## Pazarlama hunisi

| Aşama | Deneyim | Amaç | Sonraki eylem |
| --- | --- | --- | --- |
| Farkındalık | Hero, AML inceleme problemi | ARGUS'un hangi iş yükü sorununa odaklandığını anlatmak | `Explore ARGUS` |
| Ürünü anlama | Before/With ARGUS, ürün önizlemesi, çalışma akışı | Önceliklendirme, ağ bağlamı ve insan kararını açıklamak | Case Challenge |
| Etkileşimli deneyim | Sentetik Case Challenge ve kapasite hesaplayıcısı | Bağlamın inceleme sorusunu nasıl değiştirdiğini ve iş yükünü görünür kılmak | `Open Demo` |
| Ürün deneyimi | Demo girişi ve mevcut analist portalı | Overview, Investigations, Case Investigator ve Model Evidence akışını göstermek | Pilot Program |
| Pilot fırsatı | Kontrollü pilot çerçevesi ve doğrulamalı form | Kuruma özel bir değerlendirme görüşmesi için niyet toplamak | `Request ARGUS Pilot` |

`Explore ARGUS` sayfadaki etkileşimli deneyime ilerler. Case Challenge ve hesaplayıcı CTA'ları
mevcut demo girişine yönlenir. `Request ARGUS Pilot`, yalnızca oturum içi form akışını açar.

## Etkileşimli bileşen sözleşmeleri

### Case Challenge

- İşlem, hesaplar, geçmiş ve ağ bütünüyle sentetiktir.
- Senaryo açıkça “illustrative” olarak etiketlenir ve gerçek bir soruşturmayı temsil etmez.
- `YES` veya `NO` seçimi doğru/yanlış hükmü ya da suç tespiti üretmez.
- Ek geçmiş ve ağ görünümü yalnızca inceleme bağlamının değerini anlatır.
- Senaryo bilimsel model skoru veya frozen değerlendirme kanıtı olarak kullanılamaz.

### AML inceleme kapasitesi hesaplayıcısı

```text
tahmini kapasite = analist sayısı × analist başına aylık çalışma saati × 60 / ortalama vaka süresi
potansiyel açık = max(0, aylık alarm sayısı - tahmini kapasite)
```

Sonuçlar yalnızca ziyaretçinin düzenleyebildiği girdilerden türetilen aritmetik iş yükü tahminidir.
Varsayılan değerler açıklayıcıdır; ARGUS performansı, tasarruf, false-positive azalması, ROI veya
garantili verimlilik artışı ölçmez.

### Pilot talep formu

Form; ad, şirket, iş e-postası, rol, kuruluş türü ve temel zorluk alanını doğrular; mesaj alanı
isteğe bağlıdır. Değerler aktif Streamlit oturumunda tutulur. Harici CRM, e-posta teslimi veya
kalıcı veri tabanı entegrasyonu yoktur; hiçbir talep gönderilmez ve proje artifact'ları
değiştirilmez. Başarı ekranı bu sınırlamayı açıkça tekrarlar.

## Kontrollü pilot programı

Pilot, üretim kurulumu veya performans garantisi değil, yönetişim altında yürütülecek önerilen bir
değerlendirme çerçevesidir:

1. Historical Data Evaluation
2. Alert Prioritization Analysis
3. Network Intelligence Review
4. Analyst Feedback
5. Pilot Results

“Pilot Results”; gözlemleri, sınırlılıkları, analist geri bildirimini ve sonraki adım kararını ifade
eder. Önceden taahhüt edilmiş KPI artışı anlamına gelmez. Her gerçek pilot; kurumun veri erişimi,
gizlilik, güvenlik, model yönetişimi ve bağımsız doğrulama kontrollerini gerektirir.

## Dijital kampanya bileşenleri

| Bileşen | Durum | Kullanım |
| --- | --- | --- |
| Public landing page | Mevcut | Konumlandırma, problem, ürün yaklaşımı ve CTA'lar |
| Interactive Case Challenge | Mevcut | Sentetik, adım adım inceleme egzersizi |
| AML Capacity Calculator | Mevcut | Kullanıcı girdileriyle iş yükü görünümü |
| ARGUS analyst demo | Mevcut | Oturum bazlı ürün akışı |
| Pilot request form | Demo akışı | Doğrulama vardır; gönderim ve kalıcı kayıt yoktur |
| ARGUS Talks | Planlandı | Gelecekteki uzman görüşmeleri; henüz yayın veya konuşmacı iddiası yoktur |
| Case Challenges | İlk örnek mevcut | Gelecekte eklenebilecek sentetik eğitim senaryoları |
| AML Insights | Planlandı | Gelecekteki kısa eğitim içerikleri; henüz yayın iddiası yoktur |

Planlanan kartlar “planned” veya “coming soon” etiketi taşımadan yayınlanmış içerik gibi
sunulmamalıdır. Gerçek uzman, etkinlik, video, webinar, müşteri veya vaka çalışması uydurulamaz.

## Claim ve provenance sınırları

| Kaynak türü | İzin verilen kullanım | Zorunlu sınır |
| --- | --- | --- |
| Dondurulmuş bilimsel değerlendirme | Model karşılaştırması ve kayıtlı metrikler | Sentetik IBM AML HI-Small ve one-shot kronolojik holdout bağlamı belirtilir |
| İzlenen demo paketi | Arayüz ve analist iş akışını göstermek | IBM kaydı veya frozen model çıktısı değildir; bilimsel kanıt sayılmaz |
| Public Case Challenge | Pazarlama amaçlı etkileşimli örnek | Sentetik, açıklayıcı ve gerçek soruşturma olmadığı belirtilir |
| Kapasite hesaplayıcısı | Kullanıcı girdilerine dayalı iş yükü hesabı | ARGUS etkisi veya üretkenlik kazanımı olarak yorumlanmaz |
| Pilot formu | Alan doğrulama ve başarı durumu | Oturum bazlıdır; CRM, e-posta ve kalıcı kayıt yoktur |
| Pazarlama stratejisi raporu ve portfolyosu | Önerilen B2B yolculuğunu ve yaratıcı uygulamaları göstermek | Gerçekleşmiş kampanya, müşteri, kanal performansı, etkinlik veya üretim kanıtı değildir |
| Konsept görsel varlıkları | Marka, içerik, webinar ve etkinlik temaslarını örneklemek | Mockup içindeki sayı, tarih, kişi, QR kodu ve arayüz değeri doğrulanmış proje sonucu sayılmaz |

Güvenle kullanılabilecek ürün ifadeleri şunlardır:

- işlem geçmişi, strictly-prior davranışsal sinyaller ve yönlü hesap ağı bağlamıyla inceleme
  adaylarını sıralama;
- önceliklendirilmiş çalışma listesi, işlem zaman çizelgesi, hesap ağı ve model kanıtı sunma;
- gözlenen kayıtları model katkılarından görünür biçimde ayırma;
- nihai kararı eğitimli insan analiste bırakma;
- uygulama açılışında model eğitmeyip kayıtlı artifact'ları okuma.

Aşağıdaki iddialar desteklenmez:

- suç, dolandırıcılık veya kara para aklamayı kesin olarak tespit etme;
- otomatik bloke, dondurma, yaptırım, suçlama veya bildirim;
- yüzdeyle fraud/false-positive azalması, saat tasarrufu, ROI veya garantili verimlilik;
- canlı banka, gerçek müşteri, üretim başarısı, real-time operasyon veya mevzuat uyumu;
- kalibre edilmiş risk olasılığı ya da hesap seviyesinde fraud etiketi;
- GraphSAGE'i birincil operasyonel model olarak sunma.

Graph-enhanced LightGBM birincil sıralama modelidir. GraphSAGE yalnızca sınırları açıklanmış bir
araştırma karşılaştırıcısıdır. Ayrıntılar için [`MODEL_CARD.md`](../MODEL_CARD.md),
[`PROJECT_SPEC.md`](../PROJECT_SPEC.md) ve [`demo/README.md`](../../demo/README.md) kullanılmalıdır.

## Gelecek içerik kontrol listesi

Yeni bir kaynak kartı yayınlanmadan önce başlık, içerik türü, durum, tarih ve gerçek hedef URL
doğrulanmalıdır. İçerik sentetik örnek, bilimsel sonuç veya gelecek planıysa bu kapsam kartta açıkça
görünmelidir. Metrik kullanılan her içerik doğrudan dondurulmuş rapora bağlanmalı; müşteri veya
üretim sonucu izlenimi verilmemelidir.
